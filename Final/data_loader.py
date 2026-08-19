"""
Data Loading and Preprocessing Module
Handles ASAP dataset loading, cleaning, and preparation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import re
from typing import Tuple, Dict, List
from config import TRAIN_RATIO, TEST_RATIO, RANDOM_SEED
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Handles data loading, cleaning, and preprocessing for essay scoring
    """
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def load_asap_dataset(self, filepath: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        text = str(text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def normalize_scores(self, df: pd.DataFrame, score_column: str) -> pd.DataFrame:
        df_copy = df.copy()

        # Your dataset uses "set" instead of "essay_set"
        for prompt in df_copy['set'].unique():
            mask = df_copy['set'] == prompt
            df_copy.loc[mask, 'normalized_score'] = self.scaler.fit_transform(
                df_copy.loc[mask, [score_column]]
            )
        
        return df_copy
    
    def preprocess(self, filepath: str, score_column: str = 'score') -> Tuple[pd.DataFrame, Dict]:
        df = self.load_asap_dataset(filepath)

        print(df.columns)  # debug

        logger.info("Cleaning essay text...")
        df['essay'] = df['full_text'].apply(self.clean_text)

        df = df.dropna(subset=['essay', score_column])

        logger.info("Normalizing scores...")
        df = self.normalize_scores(df, score_column)

        metadata = {
            'total_samples': len(df),
            'num_prompts': df['set'].nunique(),
            'prompts': sorted(df['set'].unique()),
            'score_range': {
                prompt: (
                    int(df[df['set'] == prompt][score_column].min()),
                    int(df[df['set'] == prompt][score_column].max())
                )
                for prompt in df['set'].unique()
            }
        }
        
        logger.info(f"Preprocessing complete. Metadata: {metadata}")
        
        return df, metadata
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_df, temp_df = train_test_split(
            df,
            test_size=1 - TRAIN_RATIO,
            random_state=RANDOM_SEED,
            stratify=df['set']
        )
        
        val_df, test_df = train_test_split(
            temp_df,
            test_size=TEST_RATIO,
            random_state=RANDOM_SEED,
            stratify=temp_df['set']
        )
        
        logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df


class ASAPDataset:
    """
    PyTorch-compatible dataset class for ASAP essays
    """
    
    def __init__(self, essays: List[str], scores: np.ndarray, tokenizer, max_length: int = 512):
        self.essays = essays
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.essays)
    
    def __getitem__(self, idx: int) -> Dict:
        essay = self.essays[idx]
        score = self.scores[idx]
        
        encoding = self.tokenizer(
            text=essay,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': score
        }


# Example usage
if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    
    df, metadata = preprocessor.preprocess(
        r"D:\ESSAY SCORING\VS\ASAP_2_Final_github_train.csv",
        score_column="score"
    )
    
    train_df, val_df, test_df = preprocessor.split_data(df)
    
    print(f"Metadata: {metadata}")
    print(f"Train set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")
