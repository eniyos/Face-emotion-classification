import sys
import os
import cv2
import numpy as np
try:
    from tensorflow.keras.models import load_model
except ImportError:
    from keras.models import load_model

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.grad_cam import calculate_gradient_weighted_CAM
from utils.datasets import get_labels
from utils.inference import detect_faces, apply_offsets, load_detection_model, draw_bounding_box, load_image
from utils.preprocessor import preprocess_input

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# parameters
image_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE_DIR, 'images/test_image.jpg')
labels = get_labels('fer2013')
offsets = (0, 0)
model_filename = os.path.join(BASE_DIR, 'trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5')
detection_model_path = os.path.join(BASE_DIR, 'trained_models/detection_models/haarcascade_frontalface_default.xml')

color = (0, 255, 0)

# loading models
model = load_model(model_filename, compile=False)
target_size = model.input_shape[1:3]
face_detection = load_detection_model(detection_model_path)

# find last conv layer name
last_conv_layer = None
for layer in reversed(model.layers):
    if 'conv' in layer.name:
        last_conv_layer = layer.name
        break
if last_conv_layer is None:
    last_conv_layer = 'conv2d_7'

# loading images
rgb_image = load_image(image_path, grayscale=False)
gray_image = load_image(image_path, grayscale=True)
gray_image = np.squeeze(gray_image).astype('uint8')
faces = detect_faces(face_detection, gray_image)

# start prediction for every image
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
    rgb_image[y1:y2, x1:x2, :] = cam_resized
    draw_bounding_box((x1, y1, x2 - x1, y2 - y1), rgb_image, color)

bgr_image = cv2.cvtColor(rgb_image.astype('uint8'), cv2.COLOR_RGB2BGR)
output_path = os.path.join(BASE_DIR, 'images/guided_gradCAM.png')
cv2.imwrite(output_path, bgr_image)
print(f"Grad-CAM demo finished. Result saved to {output_path}")
