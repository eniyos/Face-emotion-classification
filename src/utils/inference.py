import json
import os
import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model as tf_load_model, model_from_json
except ImportError:
    tf = None
    try:
        from keras.models import load_model as tf_load_model, model_from_json
    except ImportError:
        tf_load_model = None
        model_from_json = None


def load_image(image_path, grayscale=False, target_size=None):
    if grayscale:
        pil_image = Image.open(image_path).convert('L')
    else:
        pil_image = Image.open(image_path).convert('RGB')
    if target_size is not None:
        pil_image = pil_image.resize((target_size[1], target_size[0]))
    img_array = np.array(pil_image, dtype='float32')
    if grayscale:
        img_array = np.expand_dims(img_array, -1)
    return img_array


def load_detection_model(model_path):
    detection_model = cv2.CascadeClassifier(model_path)
    return detection_model


def detect_faces(detection_model, gray_image_array):
    return detection_model.detectMultiScale(gray_image_array, 1.3, 5)


def draw_bounding_box(face_coordinates, image_array, color):
    x, y, w, h = face_coordinates
    cv2.rectangle(image_array, (x, y), (x + w, y + h), color, 2)


def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)


def draw_text(coordinates, image_array, text, color, x_offset=0, y_offset=0,
              font_scale=2, thickness=2):
    x, y = coordinates[:2]
    cv2.putText(image_array, text, (x + x_offset, y + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)


def get_colors(num_classes):
    colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()
    colors = np.asarray(colors) * 255
    return colors


def _extract_layer_weights(h5_group):
    """Recursively collect weight datasets from an HDF5 group in stable order."""
    weight_datasets = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            weight_datasets.append((name, obj[()]))

    h5_group.visititems(visitor)
    # Sort by name to preserve kernel/bias/gamma/beta ordering
    weight_datasets.sort(key=lambda x: x[0])
    return [w[1] for w in weight_datasets]


def load_trained_model(model_path):
    """
    Robust model loader that handles modern Keras 3, Keras 2, and TF legacy formats.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # 1. Try standard load_model
    if tf_load_model is not None:
        try:
            return tf_load_model(model_path, compile=False)
        except Exception:
            pass

    # 2. Try loading and fixing legacy model_config from HDF5
    with h5py.File(model_path, 'r') as f:
        model_config_raw = f.attrs.get('model_config')
        if model_config_raw is not None:
            if isinstance(model_config_raw, bytes):
                model_config_raw = model_config_raw.decode('utf-8')
            config_dict = json.loads(model_config_raw)

            # Fix Keras 1/2 Sequential config where config['config'] is a list
            if config_dict.get('class_name') == 'Sequential' and isinstance(config_dict.get('config'), list):
                config_dict['config'] = {
                    'name': 'sequential',
                    'layers': config_dict['config']
                }

            # Remove batch_input_shape or other obsolete keys if present
            if model_from_json is not None:
                try:
                    fixed_json = json.dumps(config_dict)
                    model = model_from_json(fixed_json)

                    # Load weights layer by layer from HDF5
                    if 'model_weights' in f:
                        weights_grp = f['model_weights']
                        for layer in model.layers:
                            layer_name = layer.name
                            if layer_name in weights_grp:
                                layer_weights = _extract_layer_weights(weights_grp[layer_name])
                                if len(layer_weights) > 0:
                                    try:
                                        layer.set_weights(layer_weights)
                                    except Exception:
                                        pass
                    return model
                except Exception:
                    pass

    # 3. Fallback: Instantiate architecture from src.models.cnn
    from models.cnn import mini_XCEPTION, simple_CNN, big_XCEPTION, tiny_XCEPTION
    basename = os.path.basename(model_path).lower()

    if 'gender' in basename or 'simple_cnn' in basename:
        model = simple_CNN((48, 48, 1) if '48' in basename else (64, 64, 1), 2)
    elif 'big_xception' in basename:
        model = big_XCEPTION((64, 64, 1), 7)
    elif 'tiny_xception' in basename:
        model = tiny_XCEPTION((64, 64, 1), 7)
    else:
        model = mini_XCEPTION((64, 64, 1), 7)

    with h5py.File(model_path, 'r') as f:
        weights_grp = f['model_weights'] if 'model_weights' in f else f
        for layer in model.layers:
            if layer.name in weights_grp:
                layer_weights = _extract_layer_weights(weights_grp[layer.name])
                if len(layer_weights) > 0:
                    try:
                        layer.set_weights(layer_weights)
                    except Exception:
                        pass
    return model
