import os
import torch

# Paths
DATASET_DIR = r"C:\projects\dog_emotion_predictor\datasets"
MODEL_SAVE_PATH = "models/dog_emotion_cnn.pth"

# Training params
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# Classes (must match folder names)
CLASSES = ["happy", "sad", "angry", "relaxed"]
NUM_CLASSES = len(CLASSES)
