"""
Main Training Script for Explainable Transformer-based AES System.

Loads an existing trained model when available, evaluates it, and automatically
scores a custom TXT, PDF, DOCX, or DOC essay from a fixed file path.
"""

import io
import logging
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import *
from data_loader import DataPreprocessor, ASAPDataset
from evaluation_metrics import AESEvaluator
from explainability import (
    SHAPExplainer,
    LIMEExplainer,
    TextLIMEExplainer,
    SHAPTextExplainer,
    ExplanationGenerator,
    train_surrogate_model,
)
from feature_extractor import LinguisticFeatureExtractor
from model_trainer import BERTAESModel, AESTrainer


RUN_EXPLAINABILITY = True
RUN_EVALUATION = True
CUSTOM_CHUNK_OVERLAP = 64

CUSTOM_ESSAY_PATH = "Essay.pdf"

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document
except ImportError:
    Document = None


os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

logging.getLogger("pypdf").setLevel(logging.ERROR)
logging.getLogger("pypdf._reader").setLevel(logging.ERROR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            os.path.join(
                LOGS_DIR,
                f"aes_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            )
        ),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


def clean_text(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_file_path(file_path):
    file_path = file_path.strip()

    if (
        len(file_path) >= 2
        and file_path[0] in ("'", '"')
        and file_path[-1] == file_path[0]
    ):
        file_path = file_path[1:-1]

    if file_path.startswith("file://"):
        file_path = file_path.replace("file://", "", 1)

    return file_path.strip()


def extract_text_from_txt_bytes(data):
    return data.decode("utf-8", errors="ignore")


def extract_text_from_pdf_bytes(data):
    if pdfplumber is not None:
        text_parts = []

        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

        text = "\n".join(text_parts).strip()

        if text:
            return text

    if PdfReader is None:
        raise ImportError("PDF support requires pypdf or pdfplumber.")

    reader = PdfReader(io.BytesIO(data), strict=False)
    text_parts = []

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)

    return "\n".join(text_parts)


def extract_text_from_docx_bytes(data):
    if Document is None:
        raise ImportError("DOCX support requires python-docx.")

    document = Document(io.BytesIO(data))
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


def extract_text_from_doc_bytes(data):
    with tempfile.NamedTemporaryFile(suffix=".doc", delete=False) as tmp_file:
        tmp_file.write(data)
        tmp_path = tmp_file.name

    try:
        result = subprocess.run(
            ["textutil", "-convert", "txt", "-stdout", tmp_path],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def extract_text_from_bytes(file_name, data):
    extension = Path(file_name).suffix.lower()

    if extension == ".txt":
        text = extract_text_from_txt_bytes(data)
    elif extension == ".pdf":
        text = extract_text_from_pdf_bytes(data)
    elif extension == ".docx":
        text = extract_text_from_docx_bytes(data)
    elif extension == ".doc":
        text = extract_text_from_doc_bytes(data)
    else:
        raise ValueError("Unsupported file type. Use .txt, .pdf, .docx, or .doc.")

    return clean_text(text)


def extract_text_from_file(file_path):
    clean_path = clean_file_path(file_path)
    path = Path(clean_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Essay file not found: {path}")

    return extract_text_from_bytes(path.name, path.read_bytes())


class AESPipeline:
    def __init__(self):
        self.device = DEVICE
        self.max_length = MAX_LENGTH

        logger.info(f"Using device: {self.device}")

        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(RANDOM_SEED)

        self.preprocessor = DataPreprocessor()
        self.feature_extractor = LinguisticFeatureExtractor()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        self.model = None
        self.trainer = None
        self.evaluator = None
        self.data_metadata = None

    def load_and_preprocess_data(self, dataset_path):
        logger.info("=" * 80)
        logger.info("STEP 1: DATA LOADING AND PREPROCESSING")
        logger.info("=" * 80)

        self.df, self.data_metadata = self.preprocessor.preprocess(dataset_path)
        self.train_df, self.val_df, self.test_df = self.preprocessor.split_data(self.df)

        logger.info(f"Training samples: {len(self.train_df)}")
        logger.info(f"Validation samples: {len(self.val_df)}")
        logger.info(f"Test samples: {len(self.test_df)}")

    def extract_features(self):
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: FEATURE EXTRACTION")
        logger.info("=" * 80)

        self.train_features = self.feature_extractor.extract_features_batch(
            self.train_df["essay"].tolist()
        )
        self.val_features = self.feature_extractor.extract_features_batch(
            self.val_df["essay"].tolist()
        )
        self.test_features = self.feature_extractor.extract_features_batch(
            self.test_df["essay"].tolist()
        )

        logger.info("Feature extraction completed.")

    def create_dataloaders(self):
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: DATALOADERS")
        logger.info("=" * 80)

        train_dataset = ASAPDataset(
            essays=self.train_df["essay"].tolist(),
            scores=self.train_df["normalized_score"].values,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )

        val_dataset = ASAPDataset(
            essays=self.val_df["essay"].tolist(),
            scores=self.val_df["normalized_score"].values,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )

        test_dataset = ASAPDataset(
            essays=self.test_df["essay"].tolist(),
            scores=self.test_df["normalized_score"].values,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )

        self.train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        self.test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        logger.info("DataLoaders created successfully.")

    def initialize_model(self):
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: MODEL INITIALIZATION")
        logger.info("=" * 80)

        self.model = BERTAESModel(model_name=MODEL_NAME).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        logger.info(f"Model: {MODEL_NAME}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

        self.trainer = AESTrainer(
            model=self.model,
            device=self.device,
            learning_rate=LEARNING_RATE,
        )

    def train_model(self):
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: MODEL TRAINING")
        logger.info("=" * 80)

        self.trainer.train(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            epochs=EPOCHS,
        )

        model_path = os.path.join(MODEL_SAVE_DIR, "bert_aes_model.pt")
        self.trainer.save_model(model_path)

        logger.info(f"Model saved to {model_path}")

    def evaluate_model(self):
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: MODEL EVALUATION")
        logger.info("=" * 80)

        self.model.eval()

        test_predictions = []
        test_labels = []

        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].float().to(self.device)

                outputs = self.model(input_ids, attention_mask)

                predictions = outputs.squeeze().detach().cpu().numpy()
                labels = labels.squeeze().detach().cpu().numpy()

                if np.isscalar(predictions):
                    predictions = np.array([predictions])

                if np.isscalar(labels):
                    labels = np.array([labels])

                test_predictions.extend(predictions.tolist())
                test_labels.extend(labels.tolist())

        test_predictions = np.asarray(test_predictions, dtype=np.float32)
        test_labels = np.asarray(test_labels, dtype=np.float32)

        self.evaluator = AESEvaluator(self.data_metadata["score_range"])

        if "set" in self.test_df.columns:
            test_prompt_ids = self.test_df["set"].values
        elif "essay_set" in self.test_df.columns:
            test_prompt_ids = self.test_df["essay_set"].values
        else:
            raise KeyError('Expected prompt column "set" or "essay_set" in test data.')

        results = self.evaluator.evaluate_all_prompts(
            test_labels,
            test_predictions,
            test_prompt_ids,
        )

        self.evaluator.print_results(results)

        return test_predictions, test_labels

    def get_available_prompts(self):
        score_range = self.data_metadata.get("score_range")

        if isinstance(score_range, dict):
            return [str(key) for key in score_range.keys()]

        if hasattr(self, "test_df"):
            if "set" in self.test_df.columns:
                return [str(value) for value in sorted(self.test_df["set"].unique())]

            if "essay_set" in self.test_df.columns:
                return [str(value) for value in sorted(self.test_df["essay_set"].unique())]

        return []

    def resolve_prompt_id(self, prompt_id=None):
        if prompt_id not in (None, "", "auto"):
            return prompt_id

        prompts = self.get_available_prompts()

        if len(prompts) == 1:
            return prompts[0]

        if "prompt_train" in prompts:
            return "prompt_train"

        if "train" in prompts:
            return "train"

        return None

    def denormalize_score(self, normalized_score, prompt_id=None):
        score_range = self.data_metadata.get("score_range")
        clipped_score = float(np.clip(normalized_score, 0.0, 1.0))
        prompt_id = self.resolve_prompt_id(prompt_id)

        selected_range = None

        if isinstance(score_range, dict):
            if prompt_id is not None:
                selected_range = score_range.get(prompt_id)

                try:
                    selected_range = selected_range or score_range.get(int(prompt_id))
                except (TypeError, ValueError):
                    pass

            if selected_range is None and len(score_range) == 1:
                selected_range = next(iter(score_range.values()))

        elif isinstance(score_range, (tuple, list)) and len(score_range) == 2:
            selected_range = score_range

        if selected_range is None:
            return None, prompt_id, None, None

        min_score, max_score = selected_range
        actual_score = clipped_score * (max_score - min_score) + min_score

        return actual_score, prompt_id, float(min_score), float(max_score)

    def predict_essay(self, essay_text):
        self.model.eval()

        token_ids = self.tokenizer(
            essay_text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        chunk_size = self.max_length - 2
        stride = max(1, chunk_size - CUSTOM_CHUNK_OVERLAP)

        if len(token_ids) <= chunk_size:
            chunks = [token_ids]
        else:
            chunks = [
                token_ids[start : start + chunk_size]
                for start in range(0, len(token_ids), stride)
            ]

        predictions = []
        pad_token_id = self.tokenizer.pad_token_id or 0

        with torch.no_grad():
            for chunk in chunks:
                input_ids = [
                    self.tokenizer.cls_token_id,
                    *chunk,
                    self.tokenizer.sep_token_id,
                ]

                attention_mask = [1] * len(input_ids)

                padding_length = self.max_length - len(input_ids)
                input_ids = input_ids + [pad_token_id] * padding_length
                attention_mask = attention_mask + [0] * padding_length

                input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
                mask_tensor = torch.tensor([attention_mask], dtype=torch.long).to(self.device)

                prediction = self.model(input_tensor, mask_tensor)
                predictions.append(float(prediction.item()))

        return {
            "raw_normalized_score": float(np.mean(predictions)),
            "clipped_normalized_score": float(np.clip(np.mean(predictions), 0.0, 1.0)),
            "token_count": len(token_ids),
            "chunk_count": len(chunks),
            "chunk_scores": predictions,
        }

    def predict_essay_score(self, essay_text, prompt_id=None):
        prediction_info = self.predict_essay(essay_text)

        actual_score, resolved_prompt_id, min_score, max_score = self.denormalize_score(
            prediction_info["clipped_normalized_score"],
            prompt_id=prompt_id,
        )

        prediction_info["actual_score"] = actual_score
        prediction_info["prompt_id"] = resolved_prompt_id
        prediction_info["min_score"] = min_score
        prediction_info["max_score"] = max_score

        return prediction_info

    def print_custom_prediction(self, file_name, essay_text, prediction_info):
        print("\n" + "=" * 80, flush=True)
        print("CUSTOM ESSAY PREDICTION", flush=True)
        print("=" * 80, flush=True)
        print(f"File: {file_name}", flush=True)
        print(f"Words extracted: {len(essay_text.split())}", flush=True)
        print(f"BERT tokens: {prediction_info['token_count']}", flush=True)
        print(f"Chunks scored: {prediction_info['chunk_count']}", flush=True)
        print(f"Prompt/set used: {prediction_info['prompt_id'] or 'auto'}", flush=True)
        print(f"Raw normalized score: {prediction_info['raw_normalized_score']:.4f}", flush=True)
        print(f"Clipped normalized score: {prediction_info['clipped_normalized_score']:.4f}", flush=True)

        if prediction_info["actual_score"] is not None:
            print(
                f"Predicted actual score: "
                f"{prediction_info['actual_score']:.2f} / {prediction_info['max_score']:.2f}",
                flush=True,
            )
            print(
                f"Original score scale: "
                f"{prediction_info['min_score']:.2f} to {prediction_info['max_score']:.2f}",
                flush=True,
            )
        else:
            print("Predicted actual score: unavailable", flush=True)

        print("\nExtracted text preview:", flush=True)
        print(essay_text[:1200], flush=True)
        print("=" * 80, flush=True)

    def score_file_path(self, file_path):
        clean_path = clean_file_path(file_path)

        print("\n" + "=" * 80, flush=True)
        print("SCORING CUSTOM ESSAY FILE", flush=True)
        print("=" * 80, flush=True)
        print(f"Using file path: {clean_path}", flush=True)

        essay_text = extract_text_from_file(clean_path)

        if not essay_text:
            raise ValueError("No readable text found in the selected file.")

        prompt_id = self.resolve_prompt_id("auto")
        prediction_info = self.predict_essay_score(essay_text, prompt_id=prompt_id)

        self.print_custom_prediction(clean_path, essay_text, prediction_info)

    def _bert_text_predict_fn(self):
        """
        Build a predict_fn(list_of_texts) -> np.ndarray of scores that runs
        essay text through the real tokenizer + BERT model. This is what the
        model actually consumes, so SHAP/LIME explanations built on top of
        this function are faithful to the deployed model.
        """
        def predict_fn(texts):
            if isinstance(texts, str):
                texts = [texts]
            texts = [str(t) for t in texts]

            self.model.eval()
            all_preds = []
            batch_size = BATCH_SIZE

            with torch.no_grad():
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    encoded = self.tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                    )
                    input_ids = encoded["input_ids"].to(self.device)
                    attention_mask = encoded["attention_mask"].to(self.device)

                    outputs = self.model(input_ids, attention_mask)
                    all_preds.append(outputs.squeeze(-1).detach().cpu().numpy())

            return np.concatenate(all_preds, axis=0) if len(all_preds) > 1 else np.atleast_1d(all_preds[0])

        return predict_fn

    def generate_explanations(self, test_predictions, test_labels):
        if not RUN_EXPLAINABILITY:
            logger.info("Explainability skipped. Set RUN_EXPLAINABILITY=True to enable.")
            return

        logger.info("\n" + "=" * 80)
        logger.info("STEP 7: EXPLAINABILITY (SHAP + LIME)")
        logger.info("=" * 80)

        try:
            text_predict_fn = self._bert_text_predict_fn()

            # ---- 1) Feature-level explainers, via a surrogate model ----
            # The BERT model only accepts tokenized text, not the linguistic
            # features (word count, readability, etc). To explain the model
            # in terms of those human-readable features, fit a surrogate
            # regressor that approximates BERT's outputs from the features,
            # then explain the surrogate with SHAP/LIME.
            n_background = min(SHAP_BACKGROUND_SAMPLES, len(self.train_df))
            background_essays = self.train_df["essay"].tolist()[:n_background]
            background_features = self.train_features.values[:n_background]
            background_predictions = text_predict_fn(background_essays)

            surrogate_model = train_surrogate_model(
                self.train_features.iloc[:n_background], background_predictions
            )

            shap_feature_explainer = SHAPExplainer(
                feature_names=self.train_features.columns.tolist(),
                background_data=background_features,
                predict_fn=surrogate_model.predict,
            )
            lime_feature_explainer = LIMEExplainer(
                feature_names=self.train_features.columns.tolist(),
                mode="regression",
                training_data=background_features,
            )

            # ---- 2) Text-level explainers, directly on the BERT model ----
            shap_text_explainer = SHAPTextExplainer(
                predict_fn=text_predict_fn, max_evals=SHAP_TEXT_MAX_EVALS
            )
            lime_text_explainer = TextLIMEExplainer(class_names=["Score"])

            generator = ExplanationGenerator(
                shap_explainer=shap_feature_explainer,
                lime_feature_explainer=lime_feature_explainer,
                lime_text_explainer=lime_text_explainer,
                shap_text_explainer=shap_text_explainer,
                feature_predict_fn=surrogate_model.predict,
                text_predict_fn=text_predict_fn,
            )

            # ---- 3) Run explanations on a handful of test essays ----
            n_samples = min(NUM_EXPLAIN_SAMPLES, len(self.test_df))
            os.makedirs(PLOTS_DIR, exist_ok=True)

            for i in range(n_samples):
                essay_text = self.test_df["essay"].iloc[i]
                features = self.test_features.iloc[i].values
                score = float(test_predictions[i])

                logger.info(f"\nExplaining test essay {i + 1}/{n_samples} "
                            f"(predicted score: {score:.4f})...")

                explanation = generator.generate_explanation(
                    essay_text=essay_text,
                    features=features,
                    score=score,
                    num_lime_samples=LIME_NUM_SAMPLES,
                    plot_dir=PLOTS_DIR,
                    sample_id=f"test_essay_{i + 1}",
                )

                logger.info(explanation["summary"])

            logger.info(f"Explainability complete. Plots saved to {PLOTS_DIR}")

        except Exception as e:
            logger.warning(f"Explainability skipped: {e}")

    def run_pipeline(self, dataset_path):
        logger.info("\n")
        logger.info("=" * 80)
        logger.info("EXPLAINABLE AUTOMATIC ESSAY SCORING SYSTEM")
        logger.info("=" * 80)

        self.load_and_preprocess_data(dataset_path)
        self.extract_features()
        self.create_dataloaders()
        self.initialize_model()

        model_path = os.path.join(MODEL_SAVE_DIR, "bert_aes_model.pt")

        if os.path.exists(model_path):
            logger.info("=" * 80)
            logger.info("LOADING TRAINED MODEL")
            logger.info("=" * 80)
            self.trainer.load_model(model_path)
        else:
            logger.info("=" * 80)
            logger.info("NO TRAINED MODEL FOUND")
            logger.info("TRAINING NEW MODEL")
            logger.info("=" * 80)
            self.train_model()

        if RUN_EVALUATION:
            test_predictions, test_labels = self.evaluate_model()
            self.generate_explanations(test_predictions, test_labels)

        self.score_file_path(CUSTOM_ESSAY_PATH)

        logger.info("\nPipeline completed successfully.")


def main():
    pipeline = AESPipeline()
    dataset_path = DATASET_PATH

    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found: {dataset_path}")
        return

    pipeline.run_pipeline(dataset_path)


if __name__ == "__main__":
    main()