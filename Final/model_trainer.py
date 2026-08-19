"""
BERT Model Training and Evaluation Module
Handles fine-tuning BERT for essay scoring
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BERTAESModel(nn.Module):
    """
    BERT-based Automatic Essay Scoring Model
    Takes essay text as input and predicts normalized score (0-1)
    """
    
    def __init__(self, model_name: str = 'bert-base-uncased', dropout_rate: float = 0.1):
        """
        Initialize the BERT-based AES model
        
        Args:
            model_name (str): Name of the pre-trained BERT model
            dropout_rate (float): Dropout rate for regularization
        """
        super(BERTAESModel, self).__init__()
        
        # Load pre-trained BERT model
        # This uses the HuggingFace transformers library
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Get BERT output dimension (usually 768 for base model)
        self.bert_hidden_size = self.bert.config.hidden_size
        
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(dropout_rate)
        
        # Linear regression head
        # Takes BERT's [CLS] token output and predicts a single score
        self.score_head = nn.Linear(self.bert_hidden_size, 1)
        
        # Sigmoid activation to bound output to [0, 1]
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model
        
        Args:
            input_ids (torch.Tensor): Token IDs from tokenizer [batch_size, seq_length]
            attention_mask (torch.Tensor): Attention mask [batch_size, seq_length]
            
        Returns:
            torch.Tensor: Predicted scores [batch_size, 1]
        """
        # Pass through BERT
        # Returns tuple of (last_hidden_state, pooled_output)
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True  # Return dict for cleaner access
        )
        
        # Get [CLS] token output (first token of each sequence)
        # This is the pooled representation of the entire sequence
        cls_token = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # Apply dropout for regularization during training
        cls_token = self.dropout(cls_token)
        
        # Pass through regression head
        score = self.score_head(cls_token)  # [batch_size, 1]
        
        # Apply sigmoid to bound output to [0, 1]
        score = self.sigmoid(score)
        
        return score


class AESTrainer:
    """
    Trainer class for BERT-based AES model
    """
    
    def __init__(self, model: BERTAESModel, device: str = 'cuda',
                 learning_rate: float = 2e-5):
        """
        Initialize trainer
        
        Args:
            model (BERTAESModel): The model to train
            device (str): Device to use ('cuda' or 'cpu')
            learning_rate (float): Learning rate for optimizer
        """
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Use MSE loss for regression
        self.criterion = nn.MSELoss()
        
        # Use AdamW optimizer (recommended for transformers)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate
        )
        
        # Learning rate scheduler for gradual decay
        self.scheduler = None
        
        # Track training history
        self.train_losses = []
        self.val_losses = []
    
    def set_scheduler(self, total_steps: int, warmup_steps: int = 500):
        """
        Set up learning rate scheduler with warmup
        
        Args:
            total_steps (int): Total training steps
            warmup_steps (int): Number of warmup steps
        """
        from transformers import get_linear_schedule_with_warmup
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch
        
        Args:
            train_loader (DataLoader): Training data loader
            
        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()  # Set model to training mode
        
        total_loss = 0
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].float().to(self.device).unsqueeze(1)  # [batch_size, 1]
            
            # Forward pass
            predictions = self.model(input_ids, attention_mask)
            
            # Calculate loss
            loss = self.criterion(predictions, labels)
            
            # Backward pass
            self.optimizer.zero_grad()  # Clear previous gradients
            loss.backward()  # Calculate gradients
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Track loss
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        
        logger.info(f"Average Training Loss: {avg_loss:.4f}")
        return avg_loss
    
    def evaluate(self, val_loader: DataLoader) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Evaluate model on validation set
        
        Args:
            val_loader (DataLoader): Validation data loader
            
        Returns:
            Tuple containing:
                - float: Average validation loss
                - np.ndarray: Predictions
                - np.ndarray: Ground truth labels
        """
        self.model.eval()  # Set model to evaluation mode
        
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        # No gradients needed for validation
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].float().to(self.device).unsqueeze(1)
                
                # Forward pass
                predictions = self.model(input_ids, attention_mask)
                
                # Calculate loss
                loss = self.criterion(predictions, labels)
                total_loss += loss.item()
                
                # Store predictions and labels
                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
        
        avg_loss = total_loss / len(val_loader)
        self.val_losses.append(avg_loss)
        
        logger.info(f"Average Validation Loss: {avg_loss:.4f}")
        
        return avg_loss, np.array(all_predictions), np.array(all_labels)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 3):
        """
        Full training loop
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            epochs (int): Number of epochs to train
            
        Returns:
            Dict: Training history
        """
        # Set up scheduler
        total_steps = len(train_loader) * epochs
        self.set_scheduler(total_steps)
        
        best_val_loss = float('inf')
        patience = 3
        patience_counter = 0
        
        for epoch in range(epochs):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            logger.info(f"{'='*50}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss, _, _ = self.evaluate(val_loader)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                logger.info(f"Validation loss improved to {val_loss:.4f}")
            else:
                patience_counter += 1
                logger.info(f"No improvement. Patience: {patience_counter}/{patience}")
                
                if patience_counter >= patience:
                    logger.info("Early stopping triggered!")
                    break
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
    
    def save_model(self, path: str):
        """Save model to disk"""
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """
        Load model from disk (compatible with CPU, CUDA and Apple Silicon MPS)
        """
        state_dict = torch.load(
            path,
            map_location=self.device
        )

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Model loaded from {path}")


def denormalize_scores(normalized_scores: np.ndarray,
                       score_range: Tuple[int, int]) -> np.ndarray:
    """
    Convert normalized scores (0-1) back to original score range
    
    Args:
        normalized_scores (np.ndarray): Scores in range [0, 1]
        score_range (Tuple[int, int]): Original score range (min, max)
        
    Returns:
        np.ndarray: Scores in original range
    """
    min_score, max_score = score_range
    denormalized = normalized_scores * (max_score - min_score) + min_score
    return np.round(denormalized).astype(int)


# Example usage
if __name__ == "__main__":
    # Initialize model
    model = BERTAESModel(model_name='bert-base-uncased')
    
    # Initialize trainer
    trainer = AESTrainer(model, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Model initialized successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
