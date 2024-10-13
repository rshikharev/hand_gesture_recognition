# Hand Gesture Recognition using MediaPipe Keypoints and Gesture Classification

This project implements a hand gesture recognition system using the MediaPipe Hands library to detect hand keypoints and a custom deep learning model to classify gestures based on these keypoints. The pipeline involves extracting hand keypoints from images in real-time and feeding them into a neural network to classify the gesture being performed (e.g., thumbs up, okay, palm, etc.).

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Models](#models)
- [Requirements](#requirements)
- [Installation](#installation)
- [Training](#training)
- [Hyperparameter Tuning with Optuna](#hyperparameter-tuning-with-optuna)
- [Inference](#inference)
- [Dataset](#dataset)
- [License](#license)

## Overview
The project uses the MediaPipe Hands library for real-time detection of 21 hand keypoints. These keypoints are then fed into a neural network (Gesture Classifier) built with PyTorch, which classifies the gesture being made.

### Key Features
- **MediaPipe Hands**: Detects 21 keypoints on the hand (including fingers, joints, and the palm).
- **Gesture Classifier**: A neural network built using PyTorch that classifies gestures based on the detected keypoints.
- **Optuna**: Integrated for hyperparameter tuning (e.g., learning rate, batch size, and dropout rate).
- **Cross-validation**: The model is trained and validated with k-fold cross-validation.
- **Logging and Monitoring**: TensorBoard is used for logging training and validation loss, accuracy, learning rate, and epoch time.
- **Confusion Matrix**: A confusion matrix is generated and saved after training to analyze the model's performance.

## Project Structure

```
hand_gesture_recognition/
│
├── models/                   # Folder to store trained models
│   └── best_gesture_classifier.pth  # Best model weights
│
├── data/                     # Folder to store datasets (images, gestures)
│   ├── train/                # Training set images
│   └── val/                  # Validation set images
│
├── src/                      # Source code for the project
│   ├── key_point.py          # Keypoint extraction and gesture classification model
│   ├── train.py              # Training script with logging and model saving
│   ├── inference.py          # Inference script for real-time gesture recognition
│   └── dataset.py            # Dataset handling and loading logic
│
├── logs/                     # TensorBoard logs
├── experiments/              # Folder for experimental results
├── requirements.txt          # Python dependencies
└── README.md                 # Project description and setup guide
```

## Models

### Gesture Classifier (PyTorch)
The gesture classifier is a neural network that takes the hand keypoints detected by MediaPipe as input and outputs a gesture class. It consists of several fully connected layers with ReLU activations and dropout to prevent overfitting. 

- **Input**: Hand keypoints (21 points, each with x, y coordinates).
- **Output**: Gesture class (e.g., thumbs up, okay, etc.).
- **Dropout**: Added to fully connected layers to prevent overfitting.
- **L2 Regularization**: Applied via weight decay in the optimizer.

### MediaPipe Hands
This is a pre-trained model from the MediaPipe library, which detects 21 keypoints on the hand. We use these keypoints as input features for the gesture classifier.

## Requirements
- Python 3.8+
- PyTorch 1.9+
- OpenCV 4.5+
- MediaPipe 0.8.3+
- numpy
- optuna
- matplotlib
- scikit-learn
- tensorboard

You can install the required dependencies using:
```bash
pip install -r requirements.txt
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/hand_gesture_recognition.git
   cd hand_gesture_recognition
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare your dataset (see the [Dataset](#dataset) section).

4. Download the pre-trained models (or train your own) and save them in the `models/` directory.

## Training

To train the gesture classification model, follow these steps:

1. Prepare your dataset in the following structure:
   ```
   data/
   ├── train/
   │   ├── class_0/   # Images for gesture class 0
   │   ├── class_1/   # Images for gesture class 1
   │   └── ...
   └── val/
       ├── class_0/   # Validation images for gesture class 0
       ├── class_1/   # Validation images for gesture class 1
       └── ...
   ```

2. Run the training script:
   ```bash
   python src/train.py
   ```

This script performs the following steps:
- Loads the dataset.
- Extracts hand keypoints from each image using MediaPipe Hands.
- Feeds the extracted keypoints into the gesture classifier.
- Logs training and validation metrics (loss, accuracy, time, etc.) using TensorBoard.
- Saves the model with the best validation loss.

You can visualize the training progress using TensorBoard:
```bash
tensorboard --logdir logs/
```

### Saving the Best Model
At the end of each epoch, the script checks the validation loss. If it is the lowest so far, the current model weights are saved to `models/best_gesture_classifier.pth`.

### Confusion Matrix
At the end of training, the script generates and saves a confusion matrix to analyze the model's performance on each class.

## Hyperparameter Tuning with Optuna
This project uses **Optuna** for automated hyperparameter tuning. The following hyperparameters are tuned:
- Learning rate (`lr`)
- Dropout rate (`dropout_rate`)
- Batch size (`batch_size`)
- Optimizer (Adam or SGD)

To start tuning with Optuna:
```bash
python src/train.py --tune
```

Optuna will run multiple trials to find the best combination of hyperparameters that minimizes the validation loss. The best model will be saved to `models/best_gesture_classifier.pth`.

## Inference

Once the model is trained, you can perform real-time inference using a webcam. The inference script loads the best model and classifies gestures based on real-time keypoint detection from MediaPipe.

Run the inference script:
```bash
python src/inference.py
```

The script will:
- Open a webcam feed.
- Detect hand keypoints using MediaPipe.
- Classify the gesture using the trained gesture classifier.
- Display the result on the video stream in real-time.

## Dataset

You will need a dataset of hand gesture images to train the classifier. The dataset should be organized into directories where each directory corresponds to a different gesture class.

Example dataset structure:
```
data/
├── train/
│   ├── class_0/   # e.g., thumbs up images
│   ├── class_1/   # e.g., thumbs down images
│   └── ...
└── val/
    ├── class_0/   # Validation images for thumbs up
    ├── class_1/   # Validation images for thumbs down
    └── ...
```

You can also create your own dataset by capturing images from a webcam and labeling them accordingly.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.