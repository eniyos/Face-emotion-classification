from scipy.io import loadmat
import pandas as pd
import numpy as np
from random import shuffle
import os
import tarfile
import zipfile
import cv2

# Project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _find_file_or_extract(dataset_dir, target_filename, archive_candidates):
    """Finds target file or extracts from archive if present."""
    target_path = os.path.join(dataset_dir, target_filename)
    if os.path.exists(target_path):
        return target_path

    # Check root of dataset_dir or subdirectories
    for root, _, files in os.walk(dataset_dir):
        if target_filename in files:
            return os.path.join(root, target_filename)

    # Check for archive to extract
    for archive_name in archive_candidates:
        archive_path = os.path.join(dataset_dir, archive_name)
        if os.path.exists(archive_path):
            print(f"Extracting {archive_path}...")
            if archive_name.endswith('.tar.gz') or archive_name.endswith('.tar'):
                with tarfile.open(archive_path, 'r:*') as tar:
                    tar.extractall(path=dataset_dir)
            elif archive_name.endswith('.zip'):
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(dataset_dir)
            # Re-check after extraction
            for root, _, files in os.walk(dataset_dir):
                if target_filename in files:
                    return os.path.join(root, target_filename)
    return target_path


class DataManager(object):
    """Class for loading fer2013 emotion classification dataset or
        imdb gender classification dataset."""
    def __init__(self, dataset_name='imdb',
                 dataset_path=None, image_size=(48, 48)):

        self.dataset_name = dataset_name
        self.image_size = image_size
        datasets_dir = os.path.join(BASE_DIR, 'datasets')

        if dataset_path is not None:
            self.dataset_path = dataset_path
        elif self.dataset_name == 'imdb':
            self.dataset_path = _find_file_or_extract(
                os.path.join(datasets_dir, 'imdb_crop'),
                'imdb.mat',
                ['imdb_crop.tar', 'imdb_crop.tar.gz', 'imdb.tar.gz']
            )
        elif self.dataset_name == 'fer2013':
            # Check in datasets/fer2013/ or datasets/
            fer_dir = os.path.join(datasets_dir, 'fer2013') if os.path.exists(os.path.join(datasets_dir, 'fer2013')) else datasets_dir
            self.dataset_path = _find_file_or_extract(
                fer_dir,
                'fer2013.csv',
                ['fer2013.tar.gz', 'fer2013.tar', 'fer2013.zip', 'challenges-in-representation-learning-facial-expression-recognition-challenge.zip']
            )
        elif self.dataset_name == 'KDEF':
            self.dataset_path = os.path.join(datasets_dir, 'KDEF')
        else:
            raise Exception(
                'Incorrect dataset name, please input imdb, fer2013 or KDEF')

    def get_data(self, usage=None):
        """
        Loads dataset.
        For fer2013, usage can be None (all), 'Training', 'PublicTest', or 'PrivateTest'.
        """
        if self.dataset_name == 'imdb':
            return self._load_imdb()
        elif self.dataset_name == 'fer2013':
            return self._load_fer2013(usage=usage)
        elif self.dataset_name == 'KDEF':
            return self._load_KDEF()

    def _load_imdb(self):
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                f"IMDB dataset not found at {self.dataset_path}. "
                "Please download imdb_crop.tar and place it in datasets/imdb_crop/"
            )
        face_score_treshold = 3
        dataset = loadmat(self.dataset_path)
        image_names_array = dataset['imdb']['full_path'][0, 0][0]
        gender_classes = dataset['imdb']['gender'][0, 0][0]
        face_score = dataset['imdb']['face_score'][0, 0][0]
        second_face_score = dataset['imdb']['second_face_score'][0, 0][0]
        face_score_mask = face_score > face_score_treshold
        second_face_score_mask = np.isnan(second_face_score)
        unknown_gender_mask = np.logical_not(np.isnan(gender_classes))
        mask = np.logical_and(face_score_mask, second_face_score_mask)
        mask = np.logical_and(mask, unknown_gender_mask)
        image_names_array = image_names_array[mask]
        gender_classes = gender_classes[mask].tolist()
        image_names = []
        for image_name_arg in range(image_names_array.shape[0]):
            image_name = image_names_array[image_name_arg][0]
            image_names.append(image_name)
        return dict(zip(image_names, gender_classes))

    def _load_fer2013(self, usage=None):
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                f"FER2013 dataset not found at {self.dataset_path}.\n"
                "To use FER2013, place 'fer2013.csv' or 'fer2013.tar.gz' inside the 'datasets/' directory.\n"
                "You can download it from Kaggle: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data"
            )
        print(f"Loading FER2013 dataset from {self.dataset_path}...")
        data = pd.read_csv(self.dataset_path)

        if usage is not None and 'Usage' in data.columns:
            data = data[data['Usage'] == usage]
            print(f"Filtered for Usage='{usage}': {len(data)} samples.")

        pixels = data['pixels'].tolist()
        width, height = 48, 48
        faces = []
        for pixel_sequence in pixels:
            face = [int(pixel) for pixel in str(pixel_sequence).split(' ')]
            face = np.asarray(face, dtype='uint8').reshape(width, height)
            if self.image_size != (width, height):
                face = cv2.resize(face, (self.image_size[1], self.image_size[0]))
            faces.append(face.astype('float32'))

        faces = np.asarray(faces)
        faces = np.expand_dims(faces, -1)

        # One-hot encode emotions 0..6
        emotions_raw = data['emotion'].values
        num_classes = 7
        emotions = np.zeros((len(emotions_raw), num_classes), dtype='float32')
        for i, val in enumerate(emotions_raw):
            emotions[i, int(val)] = 1.0

        return faces, emotions

    def _load_KDEF(self):
        class_to_arg = get_class_to_arg(self.dataset_name)
        num_classes = len(class_to_arg)

        file_paths = []
        for folder, subfolders, filenames in os.walk(self.dataset_path):
            for filename in filenames:
                if filename.lower().endswith(('.jpg')):
                    file_paths.append(os.path.join(folder, filename))

        num_faces = len(file_paths)
        y_size, x_size = self.image_size
        faces = np.zeros(shape=(num_faces, y_size, x_size))
        emotions = np.zeros(shape=(num_faces, num_classes))
        for file_arg, file_path in enumerate(file_paths):
            image_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            image_array = cv2.resize(image_array, (x_size, y_size))
            faces[file_arg] = image_array
            file_basename = os.path.basename(file_path)
            file_emotion = file_basename[4:6]
            try:
                emotion_arg = class_to_arg[file_emotion]
            except Exception:
                continue
            emotions[file_arg, emotion_arg] = 1
        faces = np.expand_dims(faces, -1)
        return faces, emotions


def get_labels(dataset_name):
    if dataset_name == 'fer2013':
        return {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
                4: 'sad', 5: 'surprise', 6: 'neutral'}
    elif dataset_name == 'imdb':
        return {0: 'woman', 1: 'man'}
    elif dataset_name == 'KDEF':
        return {0: 'AN', 1: 'DI', 2: 'AF', 3: 'HA', 4: 'SA', 5: 'SU', 6: 'NE'}
    else:
        raise Exception('Invalid dataset name')


def get_class_to_arg(dataset_name='fer2013'):
    if dataset_name == 'fer2013':
        return {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4,
                'surprise': 5, 'neutral': 6}
    elif dataset_name == 'imdb':
        return {'woman': 0, 'man': 1}
    elif dataset_name == 'KDEF':
        return {'AN': 0, 'DI': 1, 'AF': 2, 'HA': 3, 'SA': 4, 'SU': 5, 'NE': 6}
    else:
        raise Exception('Invalid dataset name')


def split_imdb_data(ground_truth_data, validation_split=.2, do_shuffle=False):
    ground_truth_keys = sorted(ground_truth_data.keys())
    if do_shuffle is not False:
        shuffle(ground_truth_keys)
    training_split = 1 - validation_split
    num_train = int(training_split * len(ground_truth_keys))
    train_keys = ground_truth_keys[:num_train]
    validation_keys = ground_truth_keys[num_train:]
    return train_keys, validation_keys


def split_data(x, y, validation_split=.2):
    num_samples = len(x)
    num_train_samples = int((1 - validation_split)*num_samples)
    train_x = x[:num_train_samples]
    train_y = y[:num_train_samples]
    val_x = x[num_train_samples:]
    val_y = y[num_train_samples:]
    train_data = (train_x, train_y)
    val_data = (val_x, val_y)
    return train_data, val_data
