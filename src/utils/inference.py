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
    """Collect weight datasets from an HDF5 group in the exact order Keras expects."""
    # Read the intended order recorded by Keras during save
    if 'weight_names' in h5_group.attrs:
        weight_names = h5_group.attrs['weight_names']
        if len(weight_names) > 0 and isinstance(weight_names[0], bytes):
            weight_names = [w.decode('utf8') for w in weight_names]

        weights = []
        for w_name in weight_names:
            # Often weight_names entries look like "conv2d_1/kernel:0" but the dataset is group['conv2d_1/kernel:0']
            # or nested. h5py group supports path access.
            if w_name in h5_group:
                weights.append(h5_group[w_name][()])
            else:
                # Some versions drop the prefix in the group hierarchy
                local_name = w_name.split('/')[-1]
                if local_name in h5_group:
                    weights.append(h5_group[local_name][()])
        if weights:
            return weights

    # Fallback to sorted (risky, but batch_norm breaks if not handled)
    weight_datasets = []
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            weight_datasets.append((name, obj[()]))

    h5_group.visititems(visitor)
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

    # Extract info from HDF5
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
                        # Older keras nested weights under group -> layer_name -> layer_name
                        for layer in model.layers:
                            layer_name = layer.name
                            if layer_name in weights_grp:
                                grp = weights_grp[layer_name]
                                # check if there is an intermediate group
                                if layer_name in grp and isinstance(grp[layer_name], h5py.Group):
                                    grp = grp[layer_name]
                                layer_weights = _extract_layer_weights(grp)
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

    # Determine input shape from the config manually
    shape = (64, 64, 1)
    try:
        with h5py.File(model_path, 'r') as f:
            cfg = f.attrs.get('model_config')
            if cfg:
                if isinstance(cfg, bytes): cfg = cfg.decode('utf-8')
                d = json.loads(cfg)
                layers = d['config'] if isinstance(d['config'], list) else d['config'].get('layers', [])
                for l in layers:
                    if 'batch_input_shape' in l.get('config', {}):
                        s = l['config']['batch_input_shape']
                        # s is usually [None, height, width, channels]
                        if len(s) == 4:
                            shape = tuple(s[1:])
                        break
    except Exception:
        pass

    if 'gender' in basename or 'simple_cnn' in basename:
        model = simple_CNN(shape, 2)
    elif 'big_xception' in basename:
        model = big_XCEPTION(shape, 7)
    elif 'tiny_xception' in basename:
        model = tiny_XCEPTION(shape, 7)
    else:
        model = mini_XCEPTION(shape, 7)

    with h5py.File(model_path, 'r') as f:
        weights_grp = f['model_weights'] if 'model_weights' in f else f
        for layer in model.layers:
            if layer.name in weights_grp:
                grp = weights_grp[layer.name]
                if layer.name in grp and isinstance(grp[layer.name], h5py.Group):
                    grp = grp[layer.name]
                layer_weights = _extract_layer_weights(grp)
                if len(layer_weights) > 0:
                    try:
                        layer.set_weights(layer_weights)
                    except Exception:
                        pass
    return model
