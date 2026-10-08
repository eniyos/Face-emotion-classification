import cv2
import numpy as np
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, load_model
except ImportError:
    tf = None
    try:
        from keras.models import Model, load_model
    except ImportError:
        pass

from .preprocessor import preprocess_input


def load_image(image_array):
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)
    return image_array


def calculate_gradient_weighted_CAM(model, preprocessed_input, layer_name='conv2d_7', category_index=None):
    if tf is None:
        raise ImportError("TensorFlow is required for Grad-CAM.")

    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(preprocessed_input)
        if category_index is None:
            category_index = tf.argmax(predictions[0])
        loss = predictions[:, category_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap.numpy(), 0)
    max_val = np.max(heatmap)
    if max_val > 0:
        heatmap /= max_val

    heatmap_resized = cv2.resize(heatmap, (64, 64))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

    img = preprocessed_input[0]
    img = img - np.min(img)
    if np.max(img) > 0:
        img = 255.0 * (img / np.max(img))
    if len(img.shape) == 2 or img.shape[-1] == 1:
        img = cv2.cvtColor(np.uint8(img), cv2.COLOR_GRAY2BGR)

    cam = np.float32(heatmap_colored) + np.float32(img)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap_resized


def deprocess_image(x):
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    x = x - x.mean()
    x = x / (x.std() + 1e-5)
    x = x * 0.1
    x = x + 0.5
    x = np.clip(x, 0, 1)
    x = x * 255
    return np.clip(x, 0, 255).astype('uint8')
