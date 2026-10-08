import os
import sys
import cv2
import numpy as np

# Suppress verbose TF logging and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.datasets import get_labels
from utils.inference import (
    draw_bounding_box, apply_offsets,
    load_detection_model, load_trained_model
)
from utils.preprocessor import preprocess_input

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model paths
detection_model_path = os.path.join(BASE_DIR, 'trained_models/detection_models/haarcascade_frontalface_default.xml')
emotion_model_path = os.path.join(BASE_DIR, 'trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5')
gender_model_path = os.path.join(BASE_DIR, 'trained_models/gender_models/gender_mini_XCEPTION.21-0.95.hdf5')

emotion_labels = get_labels('fer2013')
gender_labels = get_labels('imdb')

# Colors for emotions (BGR)
EMOTION_COLORS = {
    'angry': (0, 0, 255),      # Red
    'disgust': (0, 140, 255),  # Orange
    'fear': (128, 0, 128),     # Purple
    'happy': (0, 255, 0),      # Green
    'sad': (255, 105, 65),     # Blue
    'surprise': (0, 255, 255), # Yellow
    'neutral': (200, 200, 200) # Gray
}


def draw_hud(frame, face_coords, emotion_probs, gender_text, gender_conf, dominant_emotion, emotion_conf):
    x, y, w, h = face_coords

    # Bounding box color based on dominant emotion
    box_color = EMOTION_COLORS.get(dominant_emotion, (0, 255, 0))
    cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

    # Top header badge (Gender + Dominant Emotion)
    header_text = f"{gender_text.upper()} ({int(gender_conf*100)}%) | {dominant_emotion.upper()} ({int(emotion_conf*100)}%)"
    (tw, th), _ = cv2.getTextSize(header_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)

    header_y1 = max(0, y - th - 12)
    header_y2 = y
    cv2.rectangle(frame, (x, header_y1), (x + max(w, tw + 10), header_y2), box_color, -1)
    cv2.putText(frame, header_text, (x + 5, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    # Mini emotion probability bars on the side of bounding box
    bar_x = x + w + 10
    bar_y = y
    if bar_x + 140 < frame.shape[1]:
        # Background panel for bars
        panel_h = len(emotion_labels) * 16 + 10
        cv2.rectangle(frame, (bar_x - 5, bar_y - 5), (bar_x + 135, bar_y + panel_h), (20, 20, 20), -1)

        for idx, (emo_idx, emo_name) in enumerate(sorted(emotion_labels.items())):
            prob = emotion_probs[emo_idx]
            cur_y = bar_y + idx * 16 + 12

            # Label
            emo_col = EMOTION_COLORS.get(emo_name, (200, 200, 200))
            cv2.putText(frame, f"{emo_name[:3]}:", (bar_x, cur_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)

            # Bar
            bar_len = int(prob * 75)
            cv2.rectangle(frame, (bar_x + 35, cur_y - 8), (bar_x + 35 + bar_len, cur_y), emo_col, -1)
            cv2.rectangle(frame, (bar_x + 35, cur_y - 8), (bar_x + 110, cur_y), (80, 80, 80), 1)


def main():
    print("Loading models for webcam demo...")
    face_detection = load_detection_model(detection_model_path)
    emotion_classifier = load_trained_model(emotion_model_path)
    gender_classifier = load_trained_model(gender_model_path)

    emotion_target_size = emotion_classifier.input_shape[1:3]
    gender_target_size = gender_classifier.input_shape[1:3]

    # CLAHE for contrast normalization under uneven lighting
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Emotion tight crop (0, 0) matches FER2013 dataset; Gender uses (30, 60) for context
    gender_offsets = (30, 60)
    emotion_offsets = (0, 0)

    # State for temporal smoothing: {face_id: (smoothed_emotion_probs, smoothed_gender_probs, center_x, center_y, last_seen)}
    tracked_faces = {}
    next_face_id = 0
    alpha = 0.65  # Exponential smoothing factor (0.65 current frame, 0.35 history)

    print("Opening webcam video stream...")
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Could not open webcam. Check camera permissions.")
        return

    cv2.namedWindow('Face Emotion & Gender Classification', cv2.WINDOW_NORMAL)

    while True:
        ret, bgr_image = video_capture.read()
        if not ret or bgr_image is None:
            break

        # Flip horizontally for natural mirror feel
        bgr_image = cv2.flip(bgr_image, 1)
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

        # Detect faces with tuned parameters
        faces = face_detection.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        current_frame_centers = []

        for face_coordinates in faces:
            fx, fy, fw, fh = face_coordinates
            cx, cy = fx + fw // 2, fy + fh // 2
            current_frame_centers.append((cx, cy))

            # Match with closest tracked face or create new
            best_id = None
            min_dist = float('inf')
            for fid, data in tracked_faces.items():
                dist = np.hypot(cx - data['cx'], cy - data['cy'])
                if dist < min_dist and dist < 120:
                    min_dist = dist
                    best_id = fid

            if best_id is None:
                best_id = next_face_id
                next_face_id += 1
                tracked_faces[best_id] = {
                    'emotion_probs': np.zeros(7),
                    'gender_probs': np.zeros(2),
                    'cx': cx,
                    'cy': cy
                }

            # 1. Gender Crop & Preprocess
            gx1, gx2, gy1, gy2 = apply_offsets(face_coordinates, gender_offsets)
            gx1, gx2 = max(0, gx1), min(gray_image.shape[1], gx2)
            gy1, gy2 = max(0, gy1), min(gray_image.shape[0], gy2)

            # 2. Emotion Crop & Preprocess (Tight crop + CLAHE equalization)
            ex1, ex2, ey1, ey2 = apply_offsets(face_coordinates, emotion_offsets)
            ex1, ex2 = max(0, ex1), min(gray_image.shape[1], ex2)
            ey1, ey2 = max(0, ey1), min(gray_image.shape[0], ey2)

            try:
                # Gender face (Gray 64x64x1)
                g_crop = gray_image[gy1:gy2, gx1:gx2]
                g_crop = cv2.resize(g_crop, (gender_target_size[1], gender_target_size[0]))
                g_crop = preprocess_input(g_crop, False)
                g_crop = np.expand_dims(g_crop, (0, -1))
                gender_raw = gender_classifier.predict(g_crop, verbose=0)[0]

                # Emotion face (Gray 64x64x1 with CLAHE contrast enhancement)
                e_crop = gray_image[ey1:ey2, ex1:ex2]
                e_crop = clahe.apply(e_crop)
                e_crop = cv2.resize(e_crop, (emotion_target_size[1], emotion_target_size[0]))
                e_crop = preprocess_input(e_crop, True)
                e_crop = np.expand_dims(e_crop, (0, -1))
                emotion_raw = emotion_classifier.predict(e_crop, verbose=0)[0]
            except Exception:
                continue

            # Exponential Moving Average Smoothing
            prev_e = tracked_faces[best_id]['emotion_probs']
            prev_g = tracked_faces[best_id]['gender_probs']

            if np.all(prev_e == 0):
                smooth_e = emotion_raw
                smooth_g = gender_raw
            else:
                smooth_e = alpha * emotion_raw + (1.0 - alpha) * prev_e
                smooth_g = alpha * gender_raw + (1.0 - alpha) * prev_g

            tracked_faces[best_id]['emotion_probs'] = smooth_e
            tracked_faces[best_id]['gender_probs'] = smooth_g
            tracked_faces[best_id]['cx'] = cx
            tracked_faces[best_id]['cy'] = cy

            # Derive dominant predictions
            emo_idx = int(np.argmax(smooth_e))
            dominant_emotion = emotion_labels[emo_idx]
            emotion_conf = float(smooth_e[emo_idx])

            gen_idx = int(np.argmax(smooth_g))
            gender_text = gender_labels[gen_idx]
            gender_conf = float(smooth_g[gen_idx])

            # Draw complete HUD overlay
            draw_hud(bgr_image, face_coordinates, smooth_e, gender_text, gender_conf, dominant_emotion, emotion_conf)

        # Clean up stale tracks
        if len(faces) == 0:
            tracked_faces.clear()

        # Display instructions & FPS
        cv2.putText(bgr_image, "Press 'q' to exit | 's' to snapshot", (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('Face Emotion & Gender Classification', bgr_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            out_snap = os.path.join(BASE_DIR, 'images/webcam_snapshot.png')
            cv2.imwrite(out_snap, bgr_image)
            print(f"Snapshot saved to {out_snap}")

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
