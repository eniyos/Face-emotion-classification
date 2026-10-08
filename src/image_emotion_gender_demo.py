import sys
import os
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
    detect_faces, draw_text, draw_bounding_box,
    apply_offsets, load_detection_model, load_image,
    load_trained_model
)
from utils.preprocessor import preprocess_input

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# parameters for loading data and images
image_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE_DIR, 'images/test_image.jpg')
detection_model_path = os.path.join(BASE_DIR, 'trained_models/detection_models/haarcascade_frontalface_default.xml')
emotion_model_path = os.path.join(BASE_DIR, 'trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5')
gender_model_path = os.path.join(BASE_DIR, 'trained_models/gender_models/gender_mini_XCEPTION.21-0.95.hdf5')
emotion_labels = get_labels('fer2013')
gender_labels = get_labels('imdb')

# hyper-parameters for bounding boxes shape
gender_offsets = (30, 60)
emotion_offsets = (0, 0)

# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_trained_model(emotion_model_path)
gender_classifier = load_trained_model(gender_model_path)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]
gender_target_size = gender_classifier.input_shape[1:3]
gender_channels = gender_classifier.input_shape[-1] if len(gender_classifier.input_shape) > 3 else 1
emotion_channels = emotion_classifier.input_shape[-1] if len(emotion_classifier.input_shape) > 3 else 1

# loading images
rgb_image = load_image(image_path, grayscale=False)
gray_image = load_image(image_path, grayscale=True)
gray_image = np.squeeze(gray_image).astype('uint8')

faces = detect_faces(face_detection, gray_image)
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
            gender_face = np.expand_dims(gender_face, 0)
            gender_face = np.expand_dims(gender_face, -1)
        else:
            gender_face = rgb_image[gy1:gy2, gx1:gx2]
            gender_face = cv2.resize(gender_face, (gender_target_size[1], gender_target_size[0]))
            gender_face = preprocess_input(gender_face, False)
            gender_face = np.expand_dims(gender_face, 0)

        if emotion_channels == 1:
            emotion_face = gray_image[ey1:ey2, ex1:ex2]
            emotion_face = cv2.resize(emotion_face, (emotion_target_size[1], emotion_target_size[0]))
            emotion_face = preprocess_input(emotion_face, True)
            emotion_face = np.expand_dims(emotion_face, 0)
            emotion_face = np.expand_dims(emotion_face, -1)
        else:
            emotion_face = rgb_image[ey1:ey2, ex1:ex2]
            emotion_face = cv2.resize(emotion_face, (emotion_target_size[1], emotion_target_size[0]))
            emotion_face = preprocess_input(emotion_face, True)
            emotion_face = np.expand_dims(emotion_face, 0)
    except Exception:
        continue

    gender_prediction = gender_classifier.predict(gender_face, verbose=0)
    gender_label_arg = np.argmax(gender_prediction)
    gender_text = gender_labels[gender_label_arg]

    emotion_prediction = emotion_classifier.predict(emotion_face, verbose=0)
    emotion_label_arg = np.argmax(emotion_prediction)
    emotion_text = emotion_labels[emotion_label_arg]

    if gender_text == gender_labels[0]:
        color = (0, 0, 255)
    else:
        color = (255, 0, 0)

    draw_bounding_box(face_coordinates, rgb_image, color)
    draw_text(face_coordinates, rgb_image, gender_text, color, 0, -20, 1, 2)
    draw_text(face_coordinates, rgb_image, emotion_text, color, 0, -50, 1, 2)

bgr_image = cv2.cvtColor(rgb_image.astype('uint8'), cv2.COLOR_RGB2BGR)
output_path = os.path.join(BASE_DIR, 'images/predicted_test_image.png')
cv2.imwrite(output_path, bgr_image)
print(f"Prediction complete. Result saved to {output_path}")
