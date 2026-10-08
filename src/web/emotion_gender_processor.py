import os
import sys
import logging
import cv2
try:
    from tensorflow.keras.models import load_model
except ImportError:
    from keras.models import load_model
import numpy as np

# Add src and parent to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

BASE_DIR = os.path.dirname(src_dir)

detection_model_path = os.path.join(BASE_DIR, 'trained_models/detection_models/haarcascade_frontalface_default.xml')
emotion_model_path = os.path.join(BASE_DIR, 'trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5')
gender_model_path = os.path.join(BASE_DIR, 'trained_models/gender_models/simple_CNN.81-0.96.hdf5')
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
        emotion_classifier = load_model(emotion_model_path, compile=False)
    if gender_classifier is None:
        gender_classifier = load_model(gender_model_path, compile=False)
    return face_detection, emotion_classifier, gender_classifier


def process_image(image_bytes, output_dir=None):
    try:
        face_det, emotion_cls, gender_cls = get_models()

        gender_offsets = (10, 10)
        emotion_offsets = (0, 0)

        emotion_target_size = emotion_cls.input_shape[1:3]
        gender_target_size = gender_cls.input_shape[1:3]

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
            x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
            rgb_face = rgb_image[y1:y2, x1:x2]

            x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]

            try:
                rgb_face = cv2.resize(rgb_face, (gender_target_size[1], gender_target_size[0]))
                gray_face = cv2.resize(gray_face, (emotion_target_size[1], emotion_target_size[0]))
            except Exception:
                continue

            rgb_face = preprocess_input(rgb_face, False)
            rgb_face = np.expand_dims(rgb_face, 0)
            gender_prediction = gender_cls.predict(rgb_face)
            gender_label_arg = np.argmax(gender_prediction)
            gender_text = gender_labels[gender_label_arg]

            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_label_arg = np.argmax(emotion_cls.predict(gray_face))
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
