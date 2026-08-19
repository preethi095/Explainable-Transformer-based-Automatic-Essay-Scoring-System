# Explainable Transformer-based Automatic Essay Scoring (AES) System

## 📋 Project Overview

This project implements an explainable transformer-based system for automatic essay scoring. It combines:
- **BERT** for accurate essay scoring
- **SHAP & LIME** for generating human-readable explanations
- **Linguistic Features** for interpretability
- **Comprehensive Evaluation** metrics (QWK, RMSE, MAE)

The system not only predicts scores but also explains what linguistic features contributed to each score, making it valuable for educators and students.

---

## 📊 Dataset: ASAP Automated Essay Scoring

### Dataset Information

The **ASAP (Automated Student Assessment Prize) Automated Essay Scoring Dataset** contains:
- **12,978 essays** across **8 prompts**
- Essays written by students in **grades 7-10**
- **Human-assigned scores** for each essay
- Score ranges vary by prompt (typically 0-12 to 0-60)

### Dataset Structure

```
Column Name          | Description
--------------------|--------------------------------------------------
essay_id            | Unique identifier for each essay
essay_set           | Prompt/rubric ID (1-8)
essay               | Raw essay text
domain1_score       | Domain 1 score (primary target)
domain2_score       | Domain 2 score (if applicable)
```

### Download Instructions

**Option 1: Kaggle (Recommended)**
1. Go to: https://www.kaggle.com/c/asap-aes/data
2. Download the training data file: `training_set_rel3.tsv`
3. Place in `./data/asap_dataset.csv`

**Option 2: Direct GitHub**
1. Repository: https://github.com/fosfrancesco/asap-dataset
2. Download the CSV file
3. Save to project directory

### Prepare Dataset

```bash
# Convert TSV to CSV if needed
python prepare_dataset.py
```

---

## 🚀 Quick Start Guide

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv aes_env
source aes_env/bin/activate  # On Windows: aes_env\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Download ASAP Dataset

1. Visit Kaggle ASAP competition page
2. Download `training_set_rel3.tsv`
3. Create `./data` directory and place file there
4. Rename to `asap_dataset.csv`

### 3. Run the System

```bash
# Run complete pipeline
python main.py
```

The system will:
1. Load and preprocess data
2. Extract linguistic features
3. Train BERT model
4. Evaluate on test set
5. Generate SHAP/LIME explanations
6. Save results and models

---

## 📂 Project Structure

```
project_root/
│
├── config.py                    # Configuration settings
├── data_loader.py              # Data loading and preprocessing
├── feature_extractor.py        # Linguistic feature extraction
├── model_trainer.py            # BERT model and training
├── evaluation_metrics.py       # Evaluation metrics (QWK, RMSE, MAE)
├── explainability.py           # SHAP and LIME implementations
├── main.py                     # Main execution script
│
├── data/
│   └── asap_dataset.csv        # ASAP dataset (download needed)
│
├── models/
│   └── bert_aes_model.pt       # Saved trained model
│
├── results/
│   └── evaluation_results.json # Evaluation metrics
│
├── plots/
│   └── explanations/           # Visualization plots
│
├── logs/
│   └── aes_system_*.log        # Execution logs
│
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🔑 Key Components Explained

### 1. **data_loader.py** - Data Preprocessing
```python
# Main functions:
- load_asap_dataset()      # Load CSV file
- clean_text()             # Remove HTML, normalize whitespace
- normalize_scores()       # Scale scores to [0, 1]
- split_data()            # Train/val/test split
```

**Why it matters:** 
- Ensures consistent data quality
- Normalizes across different score ranges
- Maintains stratification by prompt

---

### 2. **feature_extractor.py** - Linguistic Features

Extracts 15 interpretable features:

| Category | Features |
|----------|----------|
| **Surface** | Word count, sentence count, avg sentence length |
| **Complexity** | Type-token ratio, complex word count, vocabulary richness |
| **Readability** | Flesch-Kincaid, Flesch Reading Ease, Gunning Fog |
| **Lexical** | Unique words, pronouns, adverbs |

**Why it matters:**
- Provides human-interpretable explanations
- Each feature correlates with writing quality
- Enables SHAP to identify which aspects influenced the score

---

### 3. **model_trainer.py** - BERT Fine-tuning

```python
class BERTAESModel(nn.Module):
    """
    Architecture:
    Input Text → [Tokenizer] → BERT → [CLS] token → Dropout → Linear → Sigmoid → Score (0-1)
    """
```

**Key Features:**
- Uses `bert-base-uncased` checkpoint
- Fine-tunes on essay scoring task
- Includes dropout for regularization
- Sigmoid output bounds predictions to [0, 1]

**Training Details:**
- Optimizer: AdamW (recommended for transformers)
- Loss function: Mean Squared Error (MSE)
- Learning rate schedule: Linear warmup + decay
- Early stopping to prevent overfitting

---

### 4. **evaluation_metrics.py** - Assessment

#### **Quadratic Weighted Kappa (QWK)**
- Standard metric in AES research
- Accounts for ordinal nature of scores
- Weights disagreements quadratically
- Range: 0-1 (higher is better)

#### **RMSE & MAE**
- RMSE: Penalizes large errors more
- MAE: Average absolute error
- Both help understand prediction distribution

---

### 5. **explainability.py** - SHAP & LIME

#### **SHAP (SHapley Additive exPlanations)**
```
For each feature:
    ├─ Calculate contribution to prediction
    ├─ Use Shapley values from game theory
    └─ Generate positive/negative impact

Result: Why specific features increased/decreased score
```

#### **LIME (Local Interpretable Model-Agnostic Explanations)**
```
For each prediction:
    ├─ Create perturbed samples around instance
    ├─ Fit simple interpretable model
    └─ Identify most influential features locally

Result: Which words/features matter for THIS essay
```

---

## 🔧 Configuration (config.py)

Key settings you might want to adjust:

```python
# Model settings
MODEL_NAME = 'bert-base-uncased'    # Change to RoBERTa if desired
MAX_LENGTH = 512                     # BERT max token limit
BATCH_SIZE = 16                      # Increase for faster training (if GPU memory allows)
EPOCHS = 3                           # Increase for better performance
LEARNING_RATE = 2e-5                # Standard for BERT fine-tuning

# Explainability
SHAP_BACKGROUND_SAMPLES = 100       # More samples = more accurate but slower
LIME_NUM_SAMPLES = 1000             # Perturbed samples for LIME

# Hardware
USE_CUDA = True                      # Set False if no GPU
```

---

## 📈 Running the System: Step-by-Step

### Step 1: Load & Preprocess Data
```
STEP 1: DATA LOADING AND PREPROCESSING
├─ Load ASAP dataset
├─ Clean essay text (remove HTML, normalize whitespace)
├─ Remove null values
├─ Normalize scores to [0, 1] range
└─ Split: 80% train, 10% val, 10% test
```

**Output:** Preprocessed essays with normalized scores

---

### Step 2: Extract Features
```
STEP 2: LINGUISTIC FEATURE EXTRACTION
├─ For each essay:
│  ├─ Count words, sentences
│  ├─ Calculate readability scores
│  ├─ Compute vocabulary richness
│  └─ Extract complexity measures
└─ Create feature matrix [num_essays × 15 features]
```

**Output:** Interpretable feature matrix

---

### Step 3: Create DataLoaders
```
STEP 3: DATALOADER CREATION
├─ Tokenize essays with BERT tokenizer
├─ Create PyTorch Dataset objects
└─ Create DataLoaders for batch training
```

**Output:** Ready-to-use training batches

---

### Step 4: Initialize Model
```
STEP 4: MODEL INITIALIZATION
├─ Load BERT from HuggingFace
├─ Add regression head (Linear layer)
├─ Move to GPU/CPU
└─ Print model statistics
```

**Example Output:**
```
Total parameters: 109,483,009
Trainable parameters: 109,483,009
```

---

### Step 5: Train Model
```
STEP 5: MODEL TRAINING
├─ For each epoch:
│  ├─ Training:
│  │  ├─ Forward pass
│  │  ├─ Calculate MSE loss
│  │  ├─ Backward pass
│  │  └─ Update weights
│  └─ Validation:
│     └─ Evaluate on validation set
└─ Early stopping if no improvement
```

**Progress indicators:** Training loss, validation loss, epoch time

---

### Step 6: Evaluate
```
STEP 6: MODEL EVALUATION
├─ Run model on test set
├─ Calculate metrics:
│  ├─ QWK (for each prompt + overall)
│  ├─ RMSE
│  ├─ MAE
│  └─ R²
└─ Print results table
```

**Example Output:**
```
================================================================================
EVALUATION RESULTS
================================================================================
     Prompt     QWK      RMSE      MAE       R²    Count
prompt_1     0.8234   0.5123   0.3421   0.7821    1234
prompt_2     0.8567   0.4532   0.2987   0.8123    1456
...
overall      0.8412   0.4821   0.3102   0.8012   12978
================================================================================
```

---

### Step 7: Generate Explanations
```
STEP 7: EXPLAINABILITY GENERATION
├─ Initialize SHAP explainer with background samples
├─ For each test sample:
│  ├─ SHAP: Calculate feature contributions
│  ├─ LIME: Identify influential words
│  └─ Generate human-readable summary
└─ Save visualizations
```

**Example Output:**
```
Sample 1:
  Predicted Score: 0.782
  True Score: 0.750
  
  Key Factors:
  • word_count: 245.0 (increased score)
  • flesch_kincaid_grade: 8.5 (increased score)
  • type_token_ratio: 0.65 (increased score)
  • complex_word_ratio: 0.18 (decreased score)
```

---

## 📊 Understanding Output Metrics

### **Quadratic Weighted Kappa (QWK)**
- **Range:** 0 to 1
- **Interpretation:**
  - 0.80-1.00: Excellent agreement
  - 0.60-0.80: Substantial agreement
  - 0.40-0.60: Moderate agreement
  - < 0.40: Fair/Poor agreement

### **RMSE (Root Mean Square Error)**
- **What it measures:** Penalizes large errors
- **Lower is better**
- **Interpretation:** Average error in score points

### **MAE (Mean Absolute Error)**
- **What it measures:** Average absolute difference
- **Lower is better**
- **More interpretable than RMSE for practical purposes**

---

## 🤖 Using the Trained Model

### Make Predictions on New Essays

```python
from transformers import AutoTokenizer
from model_trainer import BERTAESModel
import torch

# Load model
model = BERTAESModel()
model.load_state_dict(torch.load('models/bert_aes_model.pt'))
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Predict score for new essay
essay = "Your essay text here..."
encoding = tokenizer.encode_plus(
    essay,
    max_length=512,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

with torch.no_grad():
    score = model(
        encoding['input_ids'],
        encoding['attention_mask']
    )

predicted_score = score.item()
print(f"Predicted Score: {predicted_score:.3f}")  # Range: 0-1
```

### Generate Explanations for New Essays

```python
from feature_extractor import LinguisticFeatureExtractor
from explainability import SHAPExplainer, ExplanationGenerator

# Extract features
extractor = LinguisticFeatureExtractor()
features = extractor.extract_all_features(essay)

# Generate explanation
# (Requires initialized SHAP explainer and prediction function)
explanation = explanation_generator.generate_explanation(
    essay, features, predicted_score
)
summary = explanation_generator.generate_human_readable_summary(explanation)
print(summary)
```

---

## 🐛 Troubleshooting

### Issue: CUDA out of memory
**Solution:** Reduce `BATCH_SIZE` in config.py (try 8 or 4)

### Issue: Dataset not found
**Solution:** Download ASAP dataset and place in `./data/` directory

### Issue: Slow training
**Solution:** 
- Reduce number of essays (sample dataset)
- Increase `BATCH_SIZE` if GPU has memory
- Use mixed precision training

### Issue: SHAP calculations are very slow
**Solution:** Reduce `SHAP_BACKGROUND_SAMPLES` in config.py

---

## 📚 Key Concepts

### **BERT (Bidirectional Encoder Representations from Transformers)**
- Pre-trained language model
- Captures contextual meaning of words
- Bidirectional: considers both left and right context
- Used as backbone for scoring

### **Fine-tuning**
- Process of adapting pre-trained model to new task
- Only updates last few layers
- Requires less data and training time than training from scratch

### **Transfer Learning**
- Leverages knowledge from pre-training on massive corpora
- Applies to domain-specific task (essay scoring)
- Key to achieving good performance with limited data

---

## 📖 References

1. **BERT Paper:** Devlin et al., 2019 - "BERT: Pre-training of Deep Bidirectional Transformers"
2. **SHAP Paper:** Lundberg & Lee, 2017 - "A Unified Approach to Interpreting Model Predictions"
3. **LIME Paper:** Ribeiro et al., 2016 - "Why Should I Trust You?"
4. **AES Survey:** Ramesh & Sanampudi, 2022 - "An automatic essay scoring systems review"
5. **QWK Paper:** Cohen, 1968 - "Weighted kappa: Nominal scale agreement provision for scaled disagreement"

---

## 📞 Support & Questions

For issues or questions:
1. Check the Troubleshooting section
2. Review the inline code comments
3. Check log files in `./logs/` directory
4. Refer to original paper and documentation

---

## 📝 License & Attribution

This project is based on research in Automatic Essay Scoring and Explainable AI.

**Dataset Attribution:** The Hewlett Foundation, ASAP Automated Essay Scoring Dataset (2012)

---

## 🎯 Next Steps & Future Work

1. **Fine-tune on single prompt** for higher performance
2. **Experiment with RoBERTa** for potentially better results
3. **Implement multi-task learning** for multiple essay dimensions
4. **Deploy as web application** for real-world use
5. **Conduct user studies** to validate explanation usefulness
6. **Add teacher feedback integration** for active learning

---

**Last Updated:** July 2025
**Version:** 1.0
