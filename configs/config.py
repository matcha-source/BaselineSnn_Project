"""
Application configuration.

This module contains all configurable parameters used by the project.
Keeping them in one place makes experiments reproducible and avoids
hard-coding values throughout the code.
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directory where datasets will be stored
DATASET_ROOT = PROJECT_ROOT / "datasets"

# SNN Architecture parameters
INPUT_SIZE = 28 * 28
HIDDEN_SIZE = 128
OUTPUT_SIZE = 10

# LIF Parameters
BETA = 0.9
THRESHOLD = 1.0
SURROGATE_SLOPE = 25.0

# MNIST image size
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28

# Number of classes
NUM_CLASSES = 10

#Training parameters
LEARNING_RATE = 0.001
EPOCHS = 5

# Training parameters (used later)
BATCH_SIZE = 128
NUM_WORKERS = 2

# Reproducibility
RANDOM_SEED = 42

# SNN simulation parameter
TIME_STEPS = 28