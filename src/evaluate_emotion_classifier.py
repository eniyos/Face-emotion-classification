#!/usr/bin/env python3
"""
Evaluate Emotion Classifier on FER2013 test dataset.
"""

import argparse
import os
import sys
import numpy as np

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.datasets import DataManager, get_labels
from utils.inference import load_trained_model
from utils.preprocessor import preprocess_input

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def evaluate(model_path=None, dataset_path=None, usage='PrivateTest'):
    if model_path is None:
        model_path = os.path.join(BASE_DIR, 'trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5')

    print(f"Loading model from {model_path}...")
    model = load_trained_model(model_path)
    input_shape = model.input_shape[1:3]

    print(f"Loading FER2013 split '{usage}'...")
    data_loader = DataManager('fer2013', dataset_path=dataset_path, image_size=input_shape)
    faces, emotions = data_loader.get_data(usage=usage)
    faces = preprocess_input(faces)

    print(f"Running inference on {len(faces)} test images...")
    predictions = model.predict(faces, batch_size=64, verbose=1)

    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(emotions, axis=1)

    labels_map = get_labels('fer2013')
    accuracy = np.mean(y_pred == y_true) * 100.0

    print("\n" + "=" * 50)
    print(f"FER2013 Evaluation Results ({usage})")
    print("=" * 50)
    print(f"Total Test Samples: {len(faces)}")
    print(f"Overall Accuracy  : {accuracy:.2f}%\n")

    print(f"{'Class':<12} {'Emotion':<12} {'Correct/Total':<16} {'Accuracy':<10}")
    print("-" * 50)
    for class_idx in range(7):
        mask = (y_true == class_idx)
        total = np.sum(mask)
        if total > 0:
            correct = np.sum((y_pred == class_idx) & mask)
            cls_acc = (correct / total) * 100.0
            print(f"{class_idx:<12} {labels_map[class_idx]:<12} {f'{correct}/{total}':<16} {cls_acc:.2f}%")
    print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Emotion Classifier on FER2013")
    parser.add_argument('--model-path', type=str, default=None, help="Path to trained .hdf5 model")
    parser.add_argument('--dataset-path', type=str, default=None, help="Path to fer2013.csv")
    parser.add_argument('--usage', type=str, default='PrivateTest', choices=['Training', 'PublicTest', 'PrivateTest'],
                        help="Dataset split to evaluate on (default: PrivateTest)")
    args = parser.parse_args()

    evaluate(model_path=args.model_path, dataset_path=args.dataset_path, usage=args.usage)
