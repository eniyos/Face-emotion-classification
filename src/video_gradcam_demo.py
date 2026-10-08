import os
import sys
import cv2
import numpy as np
try:
    from tensorflow.keras.models import load_model
except ImportError:
    from keras.models import load_model

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.grad_cam import calculate_gradient_weighted_CAM
from utils.inference import detect_faces, apply_offsets, load_detection_model, draw_bounding_box
from utils.preprocessor import preprocess_input

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_filename = os.path.join(BASE_DIR, 'trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5')
detection_model_path = os.path.join(BASE_DIR, 'trained_models/detection_models/haarcascade_frontalface_default.xml')

model = load_model(model_filename, compile=False)
face_detection = load_detection_model(detection_model_path)
color = (0, 255, 0)
target_size = model.input_shape[1:3]
offsets = (0, 0)

last_conv_layer = None
for layer in reversed(model.layers):
    if 'conv' in layer.name:
        last_conv_layer = layer.name
        break
if last_conv_layer is None:
    last_conv_layer = 'conv2d_7'

cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)

while True:
    ret, bgr_image = video_capture.read()
    if not ret or bgr_image is None:
        break
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    faces = detect_faces(face_detection, gray_image)

    for face_coordinates in faces:
        x1, x2, y1, y2 = apply_offsets(face_coordinates, offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (target_size[1], target_size[0]))
        except Exception:
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)

        predicted_class = np.argmax(model.predict(gray_face))
        cam, heatmap = calculate_gradient_weighted_CAM(model, gray_face, last_conv_layer, predicted_class)
        cam_resized = cv2.resize(cam, (x2 - x1, y2 - y1))
        try:
            rgb_image[y1:y2, x1:x2, :] = cam_resized
        except Exception:
            continue
        draw_bounding_box((x1, y1, x2 - x1, y2 - y1), rgb_image, color)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
