import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None
from PIL import Image


def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x


def _imread(image_name):
    if cv2 is not None:
        img = cv2.imread(image_name)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.open(image_name)
    return np.array(pil_img)


def _imresize(image_array, size):
    if cv2 is not None:
        # cv2.resize expects (width, height)
        if isinstance(size, (tuple, list)):
            target_size = (int(size[1]), int(size[0]))
        else:
            target_size = (int(size), int(size))
        return cv2.resize(image_array, target_size)
    pil_img = Image.fromarray(image_array.astype('uint8'))
    resized = pil_img.resize((int(size[1]), int(size[0])))
    return np.array(resized)


def to_categorical(integer_classes, num_classes=2):
    integer_classes = np.asarray(integer_classes, dtype='int')
    num_samples = integer_classes.shape[0]
    categorical = np.zeros((num_samples, num_classes))
    categorical[np.arange(num_samples), integer_classes] = 1
    return categorical
