"""
Evaluation Metrics Module
Implements metrics for evaluating AES system performance
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray,
                           min_rating: int = 0, max_rating: int = 10) -> float:
    """
    Calculate Quadratic Weighted Kappa (QWK)
    
    QWK measures agreement between predicted and true scores, accounting for:
    - Ordinal nature of scores (score 7 vs 8 is closer than 1 vs 10)
    - Disagreement is weighted quadratically
    - Standard metric in AES research
    
    Args:
        y_true (np.ndarray): Ground truth scores
        y_pred (np.ndarray): Predicted scores
        min_rating (int): Minimum possible score
        max_rating (int): Maximum possible score
        
    Returns:
        float: QWK score (0-1, higher is better)
    """
    # Ensure integer scores for QWK
    y_true = np.round(y_true).astype(int)
    y_pred = np.round(y_pred).astype(int)
    
    # Clip to valid range
    y_true = np.clip(y_true, min_rating, max_rating)
    y_pred = np.clip(y_pred, min_rating, max_rating)
    
    # Create confusion matrix
    num_ratings = max_rating - min_rating + 1
    confusion_matrix = np.zeros((num_ratings, num_ratings))
    
    for t, p in zip(y_true, y_pred):
        confusion_matrix[t - min_rating, p - min_rating] += 1
    
    # Normalize to get proportions
    confusion_matrix = confusion_matrix / len(y_true)
    
    # Create weight matrix (quadratic disagreement)
    weights = np.zeros((num_ratings, num_ratings))
    for i in range(num_ratings):
        for j in range(num_ratings):
            weights[i, j] = ((i - j) ** 2) / ((num_ratings - 1) ** 2)
    
    # Calculate observed disagreement
    observed = np.sum(confusion_matrix * weights)
    
    # Calculate expected disagreement
    expected = 0
    for i in range(num_ratings):
        for j in range(num_ratings):
            row_sum = np.sum(confusion_matrix[i, :])
            col_sum = np.sum(confusion_matrix[:, j])
            expected += weights[i, j] * row_sum * col_sum
    
    # Calculate kappa
    if expected == 1:
        kappa = 0
    else:
        kappa = 1 - (observed / expected)
    
    return kappa


def root_mean_square_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Square Error
    
    RMSE = sqrt(mean((y_true - y_pred)^2))
    
    Args:
        y_true (np.ndarray): Ground truth scores
        y_pred (np.ndarray): Predicted scores
        
    Returns:
        float: RMSE value (lower is better)
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_error_calc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error
    
    MAE = mean(|y_true - y_pred|)
    
    Args:
        y_true (np.ndarray): Ground truth scores
        y_pred (np.ndarray): Predicted scores
        
    Returns:
        float: MAE value (lower is better)
    """
    return mean_absolute_error(y_true, y_pred)


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R-squared (coefficient of determination)
    
    R² = 1 - (SS_res / SS_tot)
    Indicates proportion of variance explained by model
    
    Args:
        y_true (np.ndarray): Ground truth scores
        y_pred (np.ndarray): Predicted scores
        
    Returns:
        float: R² value (0-1, higher is better)
    """
    return r2_score(y_true, y_pred)


class AESEvaluator:
    """
    Comprehensive evaluator for AES systems
    """
    
    def __init__(self, score_ranges: Dict[int, Tuple[int, int]]):
        """
        Initialize evaluator
        
        Args:
            score_ranges (Dict): Mapping of prompt ID to (min_score, max_score)
        """
        self.score_ranges = score_ranges
        self.results = {}
    
    def evaluate_prompt(self, y_true: np.ndarray, y_pred: np.ndarray,
                       prompt_id: int) -> Dict[str, float]:
        """
        Evaluate model for a specific prompt
        
        Args:
            y_true (np.ndarray): Ground truth scores
            y_pred (np.ndarray): Predicted scores
            prompt_id (int): ID of the essay prompt
            
        Returns:
            Dict: Dictionary of metrics
        """
        if prompt_id in self.score_ranges:
            min_score, max_score = self.score_ranges[prompt_id]
        else:
            min_score = 0
            max_score = 1
        
        metrics = {
            'qwk': quadratic_weighted_kappa(y_true, y_pred, min_score, max_score),
            'rmse': root_mean_square_error(y_true, y_pred),
            'mae': mean_absolute_error_calc(y_true, y_pred),
            'r2': r_squared(y_true, y_pred),
            'count': len(y_true)
        }
        
        self.results[prompt_id] = metrics
        
        return metrics
    
    def evaluate_all_prompts(self, y_true: np.ndarray, y_pred: np.ndarray,
                            prompt_ids: np.ndarray) -> Dict:
        """
        Evaluate model across all prompts
        
        Args:
            y_true (np.ndarray): Ground truth scores
            y_pred (np.ndarray): Predicted scores
            prompt_ids (np.ndarray): Prompt ID for each sample
            
        Returns:
            Dict: Results for each prompt and overall
        """
        results = {}
        all_y_true = []
        all_y_pred = []
        
        # Evaluate each prompt
        for prompt_id in np.unique(prompt_ids):
            mask = prompt_ids == prompt_id
            prompt_y_true = y_true[mask]
            prompt_y_pred = y_pred[mask]
            
            results[f'prompt_{prompt_id}'] = self.evaluate_prompt(
                prompt_y_true, prompt_y_pred, prompt_id
            )
            
            all_y_true.extend(prompt_y_true)
            all_y_pred.extend(prompt_y_pred)
        
        # Overall evaluation
        results['overall'] = {
            'qwk': quadratic_weighted_kappa(np.array(all_y_true), np.array(all_y_pred)),
            'rmse': root_mean_square_error(np.array(all_y_true), np.array(all_y_pred)),
            'mae': mean_absolute_error_calc(np.array(all_y_true), np.array(all_y_pred)),
            'r2': r_squared(np.array(all_y_true), np.array(all_y_pred)),
            'count': len(all_y_true)
        }
        
        return results
    
    def print_results(self, results: Dict):
        """
        Print evaluation results in a formatted table
        
        Args:
            results (Dict): Results from evaluate_all_prompts
        """
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        
        # Create DataFrame for better display
        metrics_data = []
        for prompt, metrics in results.items():
            metrics_data.append({
                'Prompt': prompt,
                'QWK': f"{metrics['qwk']:.4f}",
                'RMSE': f"{metrics['rmse']:.4f}",
                'MAE': f"{metrics['mae']:.4f}",
                'R²': f"{metrics['r2']:.4f}",
                'Count': metrics['count']
            })
        
        df = pd.DataFrame(metrics_data)
        print(df.to_string(index=False))
        print("="*80)
        
        # Highlight overall results
        if 'overall' in results:
            overall = results['overall']
            print("\nOVERALL PERFORMANCE:")
            print(f"  Quadratic Weighted Kappa (QWK): {overall['qwk']:.4f}")
            print(f"  Root Mean Square Error (RMSE):  {overall['rmse']:.4f}")
            print(f"  Mean Absolute Error (MAE):      {overall['mae']:.4f}")
            print(f"  R-squared (R²):                 {overall['r2']:.4f}")
            print("="*80 + "\n")
    
    def get_summary_statistics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Get summary statistics of predictions
        
        Args:
            y_true (np.ndarray): Ground truth scores
            y_pred (np.ndarray): Predicted scores
            
        Returns:
            Dict: Summary statistics
        """
        errors = y_true - y_pred
        
        stats = {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'min_error': np.min(errors),
            'max_error': np.max(errors),
            'median_error': np.median(errors),
            'mean_abs_error': np.mean(np.abs(errors))
        }
        
        return stats


# Example usage
if __name__ == "__main__":
    # Example data
    y_true = np.array([8, 7, 9, 6, 8, 7, 9, 8])
    y_pred = np.array([7.9, 7.1, 8.8, 6.2, 7.8, 7.3, 9.1, 8.2])
    
    # Calculate metrics
    qwk = quadratic_weighted_kappa(y_true, y_pred, 0, 10)
    rmse = root_mean_square_error(y_true, y_pred)
    mae = mean_absolute_error_calc(y_true, y_pred)
    r2 = r_squared(y_true, y_pred)
    
    print(f"QWK:  {qwk:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")
