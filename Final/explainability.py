"""
Explainability Module
Implements SHAP and LIME for generating explanations of scoring decisions
"""

import shap
import lime
import lime.lime_tabular
from lime.lime_text import LimeTextExplainer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Callable
import logging
from sklearn.ensemble import RandomForestRegressor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    SHAP (SHapley Additive exPlanations) Explainer
    Explains model predictions using Shapley values from game theory
    """
    
    def __init__(self, feature_names: List[str], background_data: np.ndarray,
                 predict_fn: Callable):
        """
        Initialize SHAP explainer
        
        Args:
            feature_names (List[str]): Names of features
            background_data (np.ndarray): Background data for SHAP (subset of training data)
            predict_fn (Callable): Function that takes features and returns predictions
        """
        self.feature_names = feature_names
        self.predict_fn = predict_fn
        
        # Initialize SHAP KernelExplainer
        # This works with any model but is computationally expensive
        self.explainer = shap.KernelExplainer(
            predict_fn,
            shap.sample(background_data, min(100, len(background_data)))
        )
    
    def explain_instance(self, features: np.ndarray) -> Dict:
        """
        Generate SHAP explanation for a single instance
        
        Args:
            features (np.ndarray): Feature vector for one essay
            
        Returns:
            Dict containing:
                - shap_values: SHAP values for each feature
                - base_value: Expected model output
                - feature_names: Names of features
                - features: Input features
        """
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(features.reshape(1, -1))[0]
        base_value = self.explainer.expected_value
        
        # Create explanation dictionary
        explanation = {
            'shap_values': shap_values,
            'base_value': base_value,
            'feature_names': self.feature_names,
            'features': features,
            'predictions': self.predict_fn(features.reshape(1, -1))[0]
        }
        
        return explanation
    
    def explain_batch(self, features: np.ndarray) -> List[Dict]:
        """
        Generate SHAP explanations for multiple instances
        
        Args:
            features (np.ndarray): Feature matrix [num_essays, num_features]
            
        Returns:
            List[Dict]: List of explanations
        """
        explanations = []
        
        for i, feature_vector in enumerate(features):
            explanation = self.explain_instance(feature_vector)
            explanations.append(explanation)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Generated SHAP explanations for {i + 1}/{len(features)} instances")
        
        return explanations
    
    def get_top_features(self, explanation: Dict, top_n: int = 5) -> List[Tuple]:
        """
        Get top N features contributing to prediction (positive and negative)
        
        Args:
            explanation (Dict): Explanation from explain_instance
            top_n (int): Number of top features to return
            
        Returns:
            List[Tuple]: List of (feature_name, shap_value, feature_value)
        """
        shap_values = explanation['shap_values']
        features = explanation['features']
        feature_names = explanation['feature_names']
        
        # Create list of (feature_name, shap_value, feature_value)
        feature_importance = [
            (name, sv, fv)
            for name, sv, fv in zip(feature_names, shap_values, features)
        ]
        
        # Sort by absolute SHAP value
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return feature_importance[:top_n]
    
    def plot_explanation(self, explanation: Dict, save_path: str = None):
        """
        Visualize SHAP explanation
        
        Args:
            explanation (Dict): Explanation from explain_instance
            save_path (str): Path to save figure
        """
        shap_values = explanation['shap_values']
        features = explanation['features']
        feature_names = explanation['feature_names']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get indices sorted by absolute SHAP value
        indices = np.argsort(np.abs(shap_values))[-10:]  # Top 10
        
        # Prepare data for bar plot
        top_features = [feature_names[i] for i in indices]
        top_values = [shap_values[i] for i in indices]
        colors = ['red' if v < 0 else 'green' for v in top_values]
        
        # Plot
        ax.barh(top_features, top_values, color=colors, alpha=0.7)
        ax.set_xlabel('SHAP Value (contribution to prediction)')
        ax.set_title('Top 10 Features Contributing to Score')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig


class LIMEExplainer:
    """
    LIME (Local Interpretable Model-Agnostic Explanations) Explainer
    Explains predictions by approximating with simple interpretable model
    """
    
    def __init__(self, feature_names: List[str], class_names: List[str] = None,
                 mode: str = 'regression', training_data: np.ndarray = None):
        """
        Initialize LIME explainer

        Args:
            feature_names (List[str]): Names of features
            class_names (List[str]): Class names (for classification)
            mode (str): 'regression' or 'classification'
            training_data (np.ndarray): Representative background data used by LIME
                to learn each feature's distribution for perturbation. Passing real
                data (rather than degenerate zeros) is required for LIME to generate
                meaningful perturbed samples.
        """
        self.feature_names = feature_names
        self.mode = mode

        if training_data is None:
            # Fallback dummy data if no real background data is supplied.
            training_data = np.zeros((2, len(feature_names)))

        # Initialize LIME tabular explainer for structured features
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=training_data,
            feature_names=feature_names,
            class_names=class_names,
            mode=mode,
            random_state=42
        )
    
    def explain_instance(self, features: np.ndarray, predict_fn: Callable,
                        num_samples: int = 1000) -> Dict:
        """
        Generate LIME explanation for a single instance
        
        Args:
            features (np.ndarray): Feature vector for one essay
            predict_fn (Callable): Function that takes features and returns predictions
            num_samples (int): Number of perturbed samples to generate
            
        Returns:
            Dict containing:
                - lime_explanation: LIME explanation object
                - features: Input features
                - predictions: Model prediction
        """
        # Generate LIME explanation
        lime_explanation = self.explainer.explain_instance(
            features,
            predict_fn,
            num_samples=num_samples
        )
        
        explanation = {
            'lime_explanation': lime_explanation,
            'features': features,
            'predictions': predict_fn(features.reshape(1, -1))[0]
        }
        
        return explanation
    
    def get_as_list(self, lime_explanation) -> List[Tuple]:
        """
        Convert LIME explanation to list of (feature, weight) tuples
        
        Args:
            lime_explanation: LIME explanation object
            
        Returns:
            List[Tuple]: List of (feature_name, weight)
        """
        return lime_explanation.as_list()
    
    def plot_explanation(self, lime_explanation, save_path: str = None):
        """
        Visualize LIME explanation
        
        Args:
            lime_explanation: LIME explanation object
            save_path (str): Path to save figure
        """
        # Get explanation data
        exp_list = lime_explanation.as_list()
        features, weights = zip(*exp_list)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Determine colors based on weight sign
        colors = ['red' if w < 0 else 'green' for w in weights]
        
        # Plot
        ax.barh(list(features), list(weights), color=colors, alpha=0.7)
        ax.set_xlabel('Weight (contribution to prediction)')
        ax.set_title('LIME Explanation of Score Prediction')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig


def train_surrogate_model(features: pd.DataFrame, model_predictions: np.ndarray,
                           random_state: int = 42) -> RandomForestRegressor:
    """
    Train a simple, interpretable-friendly surrogate model that maps the
    human-readable linguistic features (word count, readability scores, etc.)
    onto the BERT model's predicted scores.

    This is necessary because the underlying BERT model only accepts token
    IDs / attention masks as input, not the engineered linguistic features.
    To explain the model's behaviour in terms of those linguistic features
    (which is what SHAPExplainer / LIMEExplainer below operate on), we fit a
    surrogate regressor that approximates the BERT model's outputs from the
    linguistic features, and explain the surrogate instead.

    Args:
        features (pd.DataFrame): Linguistic features [num_essays, num_features]
        model_predictions (np.ndarray): BERT model predictions for the same essays
        random_state (int): Random seed for reproducibility

    Returns:
        RandomForestRegressor: Fitted surrogate model
    """
    surrogate = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=random_state,
        n_jobs=-1,
    )
    surrogate.fit(features.values, model_predictions)

    fidelity = surrogate.score(features.values, model_predictions)
    logger.info(f"Surrogate model R^2 fidelity to BERT predictions: {fidelity:.4f}")

    return surrogate


class SHAPTextExplainer:
    """
    SHAP Explainer that operates directly on essay text using the BERT model.
    Uses SHAP's Partition explainer with a text masker, which works natively
    with black-box text -> score functions (no surrogate model needed).
    """

    def __init__(self, predict_fn: Callable, max_evals: int = 300):
        """
        Initialize SHAP text explainer

        Args:
            predict_fn (Callable): Function that takes a list/array of strings
                and returns a numpy array of predicted scores
            max_evals (int): Budget of model evaluations per explanation
                (higher = more accurate but slower)
        """
        self.predict_fn = predict_fn
        self.max_evals = max_evals
        self.masker = shap.maskers.Text(r"\W+")
        self.explainer = shap.Explainer(predict_fn, self.masker)

    def explain_instance(self, text: str):
        """
        Generate a SHAP explanation for a single essay's text

        Args:
            text (str): Essay text

        Returns:
            shap.Explanation: SHAP explanation object over tokens/words
        """
        return self.explainer([text], max_evals=self.max_evals)

    def get_top_words(self, shap_explanation, top_n: int = 10) -> List[Tuple]:
        """
        Get the top N words/tokens contributing to the prediction

        Args:
            shap_explanation: Output of explain_instance
            top_n (int): Number of top tokens to return

        Returns:
            List[Tuple]: List of (token, shap_value)
        """
        values = shap_explanation.values[0]
        tokens = shap_explanation.data[0]

        pairs = list(zip(tokens, values))
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)

        return pairs[:top_n]

    def plot_explanation(self, shap_explanation, save_path: str = None, top_n: int = 15):
        """
        Visualize SHAP text explanation as a horizontal bar chart of the
        top contributing words

        Args:
            shap_explanation: Output of explain_instance
            save_path (str): Path to save figure
            top_n (int): Number of top tokens to display
        """
        top_words = self.get_top_words(shap_explanation, top_n=top_n)
        top_words = top_words[::-1]  # smallest to largest for barh

        words = [w.strip() or "·" for w, _ in top_words]
        values = [v for _, v in top_words]
        colors = ['green' if v > 0 else 'red' for v in values]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(words, values, color=colors, alpha=0.7)
        ax.set_xlabel('SHAP Value (contribution to predicted score)')
        ax.set_title('Top Words/Tokens Contributing to Essay Score (SHAP)')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")

        return fig


class TextLIMEExplainer:
    """
    LIME Text Explainer
    Explains predictions based on which words/phrases are most important
    """
    
    def __init__(self, class_names: List[str] = None):
        """
        Initialize text LIME explainer
        
        Args:
            class_names (List[str]): Class names for classification
        """
        self.class_names = class_names or ['Score']
        self.explainer = LimeTextExplainer(class_names=self.class_names)
    
    def explain_instance(self, text: str, predict_fn: Callable,
                        num_samples: int = 1000) -> Dict:
        """
        Generate LIME text explanation for a single essay
        
        Args:
            text (str): Essay text
            predict_fn (Callable): Function that takes text and returns prediction
            num_samples (int): Number of perturbed samples
            
        Returns:
            Dict containing:
                - lime_explanation: LIME explanation object
                - text: Input text
                - predictions: Model prediction
        """
        # LimeTextExplainer expects a classifier-style function that returns
        # a 2D array of shape (n_samples, n_classes). Our predict_fn returns
        # a 1D array of scalar scores (regression), so wrap it to match, and
        # explain "label" (column) 0.
        def wrapped_predict_fn(texts):
            preds = np.asarray(predict_fn(texts)).reshape(-1, 1)
            return preds

        # Generate explanation
        lime_explanation = self.explainer.explain_instance(
            text,
            wrapped_predict_fn,
            labels=(0,),
            num_samples=num_samples
        )

        explanation = {
            'lime_explanation': lime_explanation,
            'text': text,
            'predictions': predict_fn(text)
        }

        return explanation
    
    def get_as_list(self, lime_explanation) -> List[Tuple]:
        """
        Get LIME text explanation as list
        
        Args:
            lime_explanation: LIME explanation object
            
        Returns:
            List[Tuple]: List of (word, weight)
        """
        return lime_explanation.as_list(label=0)
    
    def plot_explanation(self, lime_explanation, save_path: str = None):
        """
        Visualize text LIME explanation
        
        Args:
            lime_explanation: LIME explanation object
            save_path (str): Path to save figure
        """
        # Get explanation data
        exp_list = lime_explanation.as_list(label=0)
        words, weights = zip(*exp_list)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Determine colors
        colors = ['red' if w < 0 else 'green' for w in weights]
        
        # Plot
        ax.barh(list(words), list(weights), color=colors, alpha=0.7)
        ax.set_xlabel('Weight (contribution to score)')
        ax.set_title('LIME Text Explanation - Important Words/Phrases')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig


class ExplanationGenerator:
    """
    Main class that combines SHAP and LIME to generate comprehensive explanations
    """
    
    def __init__(self, shap_explainer: SHAPExplainer = None,
                 lime_feature_explainer: LIMEExplainer = None,
                 lime_text_explainer: TextLIMEExplainer = None,
                 shap_text_explainer: SHAPTextExplainer = None,
                 feature_predict_fn: Callable = None,
                 text_predict_fn: Callable = None):
        """
        Initialize explanation generator

        Args:
            shap_explainer (SHAPExplainer): SHAP explainer over linguistic features
            lime_feature_explainer (LIMEExplainer): LIME explainer over linguistic features
            lime_text_explainer (TextLIMEExplainer): LIME explainer over raw essay text
            shap_text_explainer (SHAPTextExplainer): SHAP explainer over raw essay text
            feature_predict_fn (Callable): features -> score (e.g. surrogate model's predict)
            text_predict_fn (Callable): list[str] -> score (the actual BERT model)
        """
        self.shap_explainer = shap_explainer
        self.lime_feature_explainer = lime_feature_explainer
        self.lime_text_explainer = lime_text_explainer
        self.shap_text_explainer = shap_text_explainer
        self.feature_predict_fn = feature_predict_fn
        self.text_predict_fn = text_predict_fn

    def generate_explanation(self, essay_text: str, features: np.ndarray,
                              score: float, num_lime_samples: int = 200,
                              plot_dir: str = None, sample_id: str = "essay") -> Dict:
        """
        Generate a comprehensive explanation for a single essay's score using
        every explainer that was supplied at construction time.

        Args:
            essay_text (str): The essay text
            features (np.ndarray): Extracted linguistic features for this essay
            score (float): Model-predicted score for this essay
            num_lime_samples (int): Number of perturbed samples LIME should use
            plot_dir (str): If provided, saves a plot per explainer to this directory
            sample_id (str): Identifier used in saved plot filenames

        Returns:
            Dict containing the predicted score, essay text, and a sub-dict of
            results keyed by explainer name ('shap_features', 'lime_features',
            'shap_text', 'lime_text'), plus a human-readable summary.
        """
        import os

        explanation = {
            'essay_text': essay_text,
            'predicted_score': score,
            'explanations': {}
        }

        # --- SHAP over linguistic features (via surrogate model) ---
        if self.shap_explainer is not None:
            shap_exp = self.shap_explainer.explain_instance(np.asarray(features))
            top_features = self.shap_explainer.get_top_features(shap_exp, top_n=5)
            explanation['explanations']['shap_features'] = {
                'top_features': top_features,
                'full_explanation': shap_exp,
            }
            if plot_dir:
                self.shap_explainer.plot_explanation(
                    shap_exp, save_path=os.path.join(plot_dir, f"{sample_id}_shap_features.png")
                )
                plt.close('all')

        # --- LIME over linguistic features (via surrogate model) ---
        if self.lime_feature_explainer is not None and self.feature_predict_fn is not None:
            lime_exp = self.lime_feature_explainer.explain_instance(
                np.asarray(features), self.feature_predict_fn, num_samples=num_lime_samples
            )
            explanation['explanations']['lime_features'] = {
                'as_list': self.lime_feature_explainer.get_as_list(lime_exp['lime_explanation']),
                'full_explanation': lime_exp,
            }
            if plot_dir:
                self.lime_feature_explainer.plot_explanation(
                    lime_exp['lime_explanation'],
                    save_path=os.path.join(plot_dir, f"{sample_id}_lime_features.png")
                )
                plt.close('all')

        # --- SHAP over raw essay text (directly on the BERT model) ---
        if self.shap_text_explainer is not None:
            shap_text_exp = self.shap_text_explainer.explain_instance(essay_text)
            explanation['explanations']['shap_text'] = {
                'top_words': self.shap_text_explainer.get_top_words(shap_text_exp, top_n=10),
                'full_explanation': shap_text_exp,
            }
            if plot_dir:
                self.shap_text_explainer.plot_explanation(
                    shap_text_exp, save_path=os.path.join(plot_dir, f"{sample_id}_shap_text.png")
                )
                plt.close('all')

        # --- LIME over raw essay text (directly on the BERT model) ---
        if self.lime_text_explainer is not None and self.text_predict_fn is not None:
            lime_text_exp = self.lime_text_explainer.explain_instance(
                essay_text, self.text_predict_fn, num_samples=num_lime_samples
            )
            explanation['explanations']['lime_text'] = {
                'as_list': self.lime_text_explainer.get_as_list(lime_text_exp['lime_explanation']),
                'full_explanation': lime_text_exp,
            }
            if plot_dir:
                self.lime_text_explainer.plot_explanation(
                    lime_text_exp['lime_explanation'],
                    save_path=os.path.join(plot_dir, f"{sample_id}_lime_text.png")
                )
                plt.close('all')

        explanation['summary'] = self.generate_human_readable_summary(explanation)
        return explanation
    
    def generate_human_readable_summary(self, explanation: Dict) -> str:
        """
        Generate human-readable summary of the explanation
        
        Args:
            explanation (Dict): Explanation from generate_explanation
            
        Returns:
            str: Human-readable explanation
        """
        score = explanation['predicted_score']
        summary = f"Predicted Score: {score:.2f}\n\n"
        exps = explanation.get('explanations', {})

        if 'shap_features' in exps:
            summary += "Key Linguistic Features (SHAP, via surrogate model):\n"
            for feature_name, shap_value, feature_value in exps['shap_features']['top_features'][:5]:
                direction = "increased" if shap_value > 0 else "decreased"
                summary += f"  • {feature_name}: {feature_value:.2f} ({direction} score)\n"
            summary += "\n"

        if 'lime_features' in exps:
            summary += "Key Linguistic Features (LIME, via surrogate model):\n"
            for feature_desc, weight in exps['lime_features']['as_list'][:5]:
                direction = "increased" if weight > 0 else "decreased"
                summary += f"  • {feature_desc} ({direction} score)\n"
            summary += "\n"

        if 'shap_text' in exps:
            summary += "Key Words/Phrases (SHAP, on BERT model directly):\n"
            for token, value in exps['shap_text']['top_words'][:5]:
                direction = "increased" if value > 0 else "decreased"
                summary += f"  • \"{token.strip()}\" ({direction} score)\n"
            summary += "\n"

        if 'lime_text' in exps:
            summary += "Key Words/Phrases (LIME, on BERT model directly):\n"
            for word, weight in exps['lime_text']['as_list'][:5]:
                direction = "increased" if weight > 0 else "decreased"
                summary += f"  • \"{word}\" ({direction} score)\n"

        return summary


# Example usage
if __name__ == "__main__":
    logger.info("Explainability module loaded successfully!")
