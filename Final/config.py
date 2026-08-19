"""
Configuration file for the Explainable Transformer-based AES System
Contains all hyperparameters, file paths, and settings
"""

import os

# ==================== DATASET CONFIGURATION ====================
DATASET_PATH = "ASAP_2_Final_github_train.csv" # Path to ASAP dataset
TRAIN_RATIO = 0.8  # 80% training, 20% validation/test
TEST_RATIO = 0.5   # Of remaining 20%, split into val/test equally

# ==================== MODEL CONFIGURATION ====================
MODEL_NAME = 'bert-base-uncased'  # Pre-trained BERT model
MAX_LENGTH = 512  # Maximum token length for BERT (BERT limit)
BATCH_SIZE = 16   # Batch size for training
LEARNING_RATE = 2e-5  # Learning rate for fine-tuning
EPOCHS = 3  # Number of training epochs
WARMUP_STEPS = 500  # Warmup steps for learning rate scheduler

# ==================== DEVICE CONFIGURATION ====================
import torch

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")      # Apple Silicon GPU
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")     # NVIDIA GPU
else:
    DEVICE = torch.device("cpu")      # CPU fallback

# ==================== PATHS ====================
OUTPUT_DIR = './results/'  # Directory to save results
MODEL_SAVE_DIR = './models/'  # Directory to save trained models
PLOTS_DIR = './plots/'  # Directory to save plots
LOGS_DIR = './logs/'  # Directory to save logs

# ==================== LINGUISTIC FEATURES ====================
# Features to extract from essays for explainability
LINGUISTIC_FEATURES = [
    'word_count',
    'sentence_count',
    'avg_sentence_length',
    'avg_word_length',
    'type_token_ratio',
    'flesch_kincaid_grade',
    'flesch_reading_ease',
    'gunning_fog_index',
    'smog_index',
    'dale_chall_score',
    'complex_word_count',
    'complex_word_ratio',
    'unique_word_count',
    'pronoun_count',
    'adverb_count',
]

# ==================== EXPLAINABILITY CONFIGURATION ====================
SHAP_BACKGROUND_SAMPLES = 100  # Number of background samples for SHAP
LIME_NUM_SAMPLES = 200  # Number of perturbed samples LIME generates per explanation
LIME_NUM_FEATURES = 10  # Number of features to display in LIME explanation
NUM_EXPLAIN_SAMPLES = 3  # Number of test essays to generate SHAP/LIME explanations for
SHAP_TEXT_MAX_EVALS = 200  # Max model evaluations per SHAP text explanation

# ==================== EVALUATION METRICS ====================
EVALUATION_METRICS = ['QWK', 'RMSE', 'MAE']  # Metrics to report

# ==================== RANDOM SEED ====================
RANDOM_SEED = 42  # For reproducibility

# Create directories if they don't exist
for directory in [OUTPUT_DIR, MODEL_SAVE_DIR, PLOTS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)
