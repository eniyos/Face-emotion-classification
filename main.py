#!/usr/bin/env python3
"""
Face Emotion & Gender Classification
CLI entry point for running inference, web server, training, or evaluation.
"""

import argparse
import os
import sys

# Suppress verbose TF logging and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

# Ensure src is in Python path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_DIR, 'src'))


def run_image(image_path, output_path=None):
    import cv2
    import numpy as np

    from utils.datasets import get_labels
    from utils.inference import (
        detect_faces, draw_text, draw_bounding_box,
        apply_offsets, load_detection_model, load_image,
        load_trained_model
    )
    from utils.preprocessor import preprocess_input

    detection_model_path = os.path.join(PROJECT_DIR, 'trained_models/detection_models/haarcascade_frontalface_default.xml')
    emotion_model_path = os.path.join(PROJECT_DIR, 'trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5')
    gender_model_path = os.path.join(PROJECT_DIR, 'trained_models/gender_models/gender_mini_XCEPTION.21-0.95.hdf5')

    emotion_labels = get_labels('fer2013')
    gender_labels = get_labels('imdb')

    gender_offsets = (30, 60)
    emotion_offsets = (0, 0)

    print("Loading models...")
    face_detection = load_detection_model(detection_model_path)
    emotion_classifier = load_trained_model(emotion_model_path)
    gender_classifier = load_trained_model(gender_model_path)

    emotion_target_size = emotion_classifier.input_shape[1:3]
    gender_target_size = gender_classifier.input_shape[1:3]
    gender_channels = gender_classifier.input_shape[-1] if len(gender_classifier.input_shape) > 3 else 1
    emotion_channels = emotion_classifier.input_shape[-1] if len(emotion_classifier.input_shape) > 3 else 1

    print(f"Loading image from {image_path}...")
    rgb_image = load_image(image_path, grayscale=False)
    gray_image = load_image(image_path, grayscale=True)
    gray_image = np.squeeze(gray_image).astype('uint8')

    faces = detect_faces(face_detection, gray_image)
    print(f"Detected {len(faces)} face(s).")

    for i, face_coordinates in enumerate(faces):
        # Gender face crop
        gx1, gx2, gy1, gy2 = apply_offsets(face_coordinates, gender_offsets)
        gx1, gx2 = max(0, gx1), min(rgb_image.shape[1], gx2)
        gy1, gy2 = max(0, gy1), min(rgb_image.shape[0], gy2)

        # Emotion face crop
        ex1, ex2, ey1, ey2 = apply_offsets(face_coordinates, emotion_offsets)
        ex1, ex2 = max(0, ex1), min(gray_image.shape[1], ex2)
        ey1, ey2 = max(0, ey1), min(gray_image.shape[0], ey2)

        try:
            if gender_channels == 1:
                gender_face = gray_image[gy1:gy2, gx1:gx2]
                gender_face = cv2.resize(gender_face, (gender_target_size[1], gender_target_size[0]))
                gender_face = preprocess_input(gender_face, False)
                gender_face = np.expand_dims(gender_face, 0)
                gender_face = np.expand_dims(gender_face, -1)
            else:
                gender_face = rgb_image[gy1:gy2, gx1:gx2]
                gender_face = cv2.resize(gender_face, (gender_target_size[1], gender_target_size[0]))
                gender_face = preprocess_input(gender_face, False)
                gender_face = np.expand_dims(gender_face, 0)

            if emotion_channels == 1:
                emotion_face = gray_image[ey1:ey2, ex1:ex2]
                emotion_face = cv2.resize(emotion_face, (emotion_target_size[1], emotion_target_size[0]))
                emotion_face = preprocess_input(emotion_face, True)
                emotion_face = np.expand_dims(emotion_face, 0)
                emotion_face = np.expand_dims(emotion_face, -1)
            else:
                emotion_face = rgb_image[ey1:ey2, ex1:ex2]
                emotion_face = cv2.resize(emotion_face, (emotion_target_size[1], emotion_target_size[0]))
                emotion_face = preprocess_input(emotion_face, True)
                emotion_face = np.expand_dims(emotion_face, 0)
        except Exception:
            continue

        gender_prediction = gender_classifier.predict(gender_face, verbose=0)
        gender_label_arg = np.argmax(gender_prediction)
        gender_text = gender_labels[gender_label_arg]

        emotion_prediction = emotion_classifier.predict(emotion_face, verbose=0)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]

        print(f"Face {i+1}: Gender = {gender_text} ({gender_prediction[0][gender_label_arg]:.2f}), Emotion = {emotion_text} ({emotion_prediction[0][emotion_label_arg]:.2f})")

        color = (0, 0, 255) if gender_text == gender_labels[0] else (255, 0, 0)

        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, gender_text, color, 0, -20, 1, 2)
        draw_text(face_coordinates, rgb_image, emotion_text, color, 0, -50, 1, 2)

    bgr_image = cv2.cvtColor(rgb_image.astype('uint8'), cv2.COLOR_RGB2BGR)
    if output_path is None:
        output_path = os.path.join(PROJECT_DIR, 'images/predicted_test_image.png')
    cv2.imwrite(output_path, bgr_image)
    print(f"Saved annotated output to {output_path}")


def run_webcam():
    import subprocess
    subprocess.run([sys.executable, os.path.join(PROJECT_DIR, 'src/video_emotion_gender_demo.py')])


def run_server(port):
    from web.faces import app
    print(f"Starting Face Classification API on port {port}...")
    app.run(debug=False, host='0.0.0.0', port=port)


def main():
    parser = argparse.ArgumentParser(description="Face Emotion & Gender Classification")
    parser.add_argument('--image', type=str, help="Path to input image for classification")
    parser.add_argument('--output', type=str, default=None, help="Path to save classified output image")
    parser.add_argument('--webcam', action='store_true', help="Run real-time webcam classification demo")
    parser.add_argument('--server', action='store_true', help="Start Flask classification server")
    parser.add_argument('--port', type=int, default=8084, help="Port for Flask server (default 8084)")
    parser.add_argument('--train-emotion', action='store_true', help="Train emotion classifier on FER2013")
    parser.add_argument('--eval-emotion', action='store_true', help="Evaluate emotion classifier on FER2013 test set")
    parser.add_argument('--dataset-path', type=str, default=None, help="Custom path to fer2013.csv")
    parser.add_argument('--epochs', type=int, default=100, help="Epochs for training (default: 100)")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size for training (default: 32)")
    parser.add_argument('--model-type', type=str, default='mini_XCEPTION',
                        choices=['mini_XCEPTION', 'simple_CNN', 'big_XCEPTION', 'tiny_XCEPTION'],
                        help="Model architecture for emotion training (default: mini_XCEPTION)")
    args = parser.parse_args()

    if args.server:
        run_server(args.port)
    elif args.webcam:
        run_webcam()
    elif args.train_emotion:
        from train_emotion_classifier import train
        train(
            dataset_path=args.dataset_path,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            model_type=args.model_type
        )
    elif args.eval_emotion:
        from evaluate_emotion_classifier import evaluate
        evaluate(dataset_path=args.dataset_path)
    elif args.image:
        run_image(args.image, args.output)
    else:
        default_img = os.path.join(PROJECT_DIR, 'images/test_image.jpg')
        if os.path.exists(default_img):
            print(f"No arguments provided. Running default image test on {default_img}...")
            run_image(default_img, args.output)
        else:
            parser.print_help()


if __name__ == '__main__':
    main()
