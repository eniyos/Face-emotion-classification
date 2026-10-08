#!/usr/bin/env python3
"""
Train Emotion Classification Model on FER2013 dataset.
"""

import argparse
import os
import sys

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
except ImportError:
    from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    from keras.preprocessing.image import ImageDataGenerator

from models.cnn import mini_XCEPTION, simple_CNN, big_XCEPTION, tiny_XCEPTION
from utils.datasets import DataManager, split_data
from utils.preprocessor import preprocess_input

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def train(dataset_path=None, batch_size=32, num_epochs=100, patience=20,
          model_type='mini_XCEPTION', output_dir=None):

    input_shape = (64, 64, 1)
    num_classes = 7
    validation_split = 0.2

    if output_dir is None:
        output_dir = os.path.join(BASE_DIR, 'trained_models/emotion_models')
    os.makedirs(output_dir, exist_ok=True)

    # Data Augmentation generator
    data_generator = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )

    # Build model
    print(f"Building {model_type} architecture for FER2013...")
    if model_type == 'simple_CNN':
        model = simple_CNN(input_shape, num_classes)
    elif model_type == 'big_XCEPTION':
        model = big_XCEPTION(input_shape, num_classes)
    elif model_type == 'tiny_XCEPTION':
        model = tiny_XCEPTION(input_shape, num_classes)
    else:
        model = mini_XCEPTION(input_shape, num_classes)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Load dataset
    data_loader = DataManager('fer2013', dataset_path=dataset_path, image_size=input_shape[:2])
    faces, emotions = data_loader.get_data()
    faces = preprocess_input(faces)
    num_samples = len(faces)
    print(f"Loaded {num_samples} samples from FER2013.")

    train_data, val_data = split_data(faces, emotions, validation_split)
    train_faces, train_emotions = train_data

    # Callbacks
    log_file_path = os.path.join(output_dir, 'fer2013_emotion_training.log')
    csv_logger = CSVLogger(log_file_path, append=False)
    early_stop = EarlyStopping('val_loss', patience=patience, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience / 4), verbose=1)

    model_checkpoint_path = os.path.join(
        output_dir, f'fer2013_{model_type}.{{epoch:02d}}-{{val_loss:.2f}}.hdf5'
    )
    model_checkpoint = ModelCheckpoint(model_checkpoint_path, 'val_loss', verbose=1, save_best_only=True)

    callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

    print(f"Starting training on {len(train_faces)} samples, validating on {len(val_data[0])} samples...")
    model.fit(
        data_generator.flow(train_faces, train_emotions, batch_size),
        steps_per_epoch=int(len(train_faces) / batch_size),
        epochs=num_epochs,
        verbose=1,
        callbacks=callbacks,
        validation_data=val_data
    )

    final_model_path = os.path.join(output_dir, f'fer2013_{model_type}_final.hdf5')
    model.save(final_model_path)
    print(f"Training complete! Final model saved to {final_model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Emotion Classifier on FER2013")
    parser.add_argument('--dataset-path', type=str, default=None, help="Path to fer2013.csv")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument('--epochs', type=int, default=100, help="Epochs (default: 100)")
    parser.add_argument('--patience', type=int, default=20, help="Early stopping patience (default: 20)")
    parser.add_argument('--model', type=str, default='mini_XCEPTION',
                        choices=['mini_XCEPTION', 'simple_CNN', 'big_XCEPTION', 'tiny_XCEPTION'],
                        help="Model architecture")
    args = parser.parse_args()

    train(
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        patience=args.patience,
        model_type=args.model
    )
