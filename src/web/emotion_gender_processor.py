import os
import sys
import logging
import cv2
import numpy as np

# Add src and parent to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

from utils.datasets import get_labels
from utils.inference import (
    detect_faces, draw_text, draw_bounding_box,
    apply_offsets, load_detection_model, load_trained_model
)
from utils.preprocessor import preprocess_input

BASE_DIR = os.path.dirname(src_dir)

detection_model_path = os.path.join(BASE_DIR, 'trained_models/detection_models/haarcascade_frontalface_default.xml')
emotion_model_path = os.path.join(BASE_DIR, 'trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5')
gender_model_path = os.path.join(BASE_DIR, 'trained_models/gender_models/gender_mini_XCEPTION.21-0.95.hdf5')
emotion_labels = get_labels('fer2013')
gender_labels = get_labels('imdb')

# lazy loaded models
face_detection = None
emotion_classifier = None
gender_classifier = None


def get_models():
    global face_detection, emotion_classifier, gender_classifier
    if face_detection is None:
        face_detection = load_detection_model(detection_model_path)
    if emotion_classifier is None:
        emotion_classifier = load_trained_model(emotion_model_path)
    if gender_classifier is None:
        gender_classifier = load_trained_model(gender_model_path)
    return face_detection, emotion_classifier, gender_classifier


def process_image(image_bytes, output_dir=None):
    try:
        face_det, emotion_cls, gender_cls = get_models()

        gender_offsets = (30, 60)
        emotion_offsets = (0, 0)

        emotion_target_size = emotion_cls.input_shape[1:3]
        gender_target_size = gender_cls.input_shape[1:3]
        gender_channels = gender_cls.input_shape[-1] if len(gender_cls.input_shape) > 3 else 1
        emotion_channels = emotion_cls.input_shape[-1] if len(emotion_cls.input_shape) > 3 else 1

        # loading images from bytes
        image_array = np.frombuffer(image_bytes, np.uint8)
        unchanged_image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)
        if unchanged_image is None:
            raise ValueError("Failed to decode image")

        if len(unchanged_image.shape) == 2:
            rgb_image = cv2.cvtColor(unchanged_image, cv2.COLOR_GRAY2RGB)
            gray_image = unchanged_image
        else:
            rgb_image = cv2.cvtColor(unchanged_image, cv2.COLOR_BGR2RGB)
            gray_image = cv2.cvtColor(unchanged_image, cv2.COLOR_BGR2GRAY)

        faces = detect_faces(face_det, gray_image)
        for face_coordinates in faces:
            gx1, gx2, gy1, gy2 = apply_offsets(face_coordinates, gender_offsets)
            gx1, gx2 = max(0, gx1), min(rgb_image.shape[1], gx2)
            gy1, gy2 = max(0, gy1), min(rgb_image.shape[0], gy2)

            ex1, ex2, ey1, ey2 = apply_offsets(face_coordinates, emotion_offsets)
            ex1, ex2 = max(0, ex1), min(gray_image.shape[1], ex2)
            ey1, ey2 = max(0, ey1), min(gray_image.shape[0], ey2)

            try:
                if gender_channels == 1:
                    gender_face = gray_image[gy1:gy2, gx1:gx2]
                    gender_face = cv2.resize(gender_face, (gender_target_size[1], gender_target_size[0]))
                    gender_face = preprocess_input(gender_face, False)
                    gender_face = np.expand_dims(gender_face, (0, -1))
                else:
                    gender_face = rgb_image[gy1:gy2, gx1:gx2]
                    gender_face = cv2.resize(gender_face, (gender_target_size[1], gender_target_size[0]))
                    gender_face = preprocess_input(gender_face, False)
                    gender_face = np.expand_dims(gender_face, 0)

                if emotion_channels == 1:
                    emotion_face = gray_image[ey1:ey2, ex1:ex2]
                    emotion_face = cv2.resize(emotion_face, (emotion_target_size[1], emotion_target_size[0]))
                    emotion_face = preprocess_input(emotion_face, True)
                    emotion_face = np.expand_dims(emotion_face, (0, -1))
                else:
                    emotion_face = rgb_image[ey1:ey2, ex1:ex2]
                    emotion_face = cv2.resize(emotion_face, (emotion_target_size[1], emotion_target_size[0]))
                    emotion_face = preprocess_input(emotion_face, True)
                    emotion_face = np.expand_dims(emotion_face, 0)
            except Exception:
                continue

            gender_prediction = gender_cls.predict(gender_face, verbose=0)
            gender_label_arg = np.argmax(gender_prediction)
            gender_text = gender_labels[gender_label_arg]

            emotion_prediction = emotion_cls.predict(emotion_face, verbose=0)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = emotion_labels[emotion_label_arg]

            if gender_text == gender_labels[0]:
                color = (0, 0, 255)
            else:
                color = (255, 0, 0)

            draw_bounding_box(face_coordinates, rgb_image, color)
            draw_text(face_coordinates, rgb_image, gender_text, color, 0, -20, 1, 2)
            draw_text(face_coordinates, rgb_image, emotion_text, color, 0, -50, 1, 2)

        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        if output_dir is None:
            output_dir = os.path.join(BASE_DIR, 'result')
        os.makedirs(output_dir, exist_ok=True)
        out_file = os.path.join(output_dir, 'predicted_image.png')
        cv2.imwrite(out_file, bgr_image)
        return out_file
    except Exception as err:
        logging.error(f'Error in emotion gender processor: "{err}"')
        raise
