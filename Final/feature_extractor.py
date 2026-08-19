"""
Linguistic Feature Extraction Module
Extracts interpretable linguistic features from essays for explainability
"""

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import textstat
import numpy as np
import pandas as pd
from typing import Dict, List, Union
import logging

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LinguisticFeatureExtractor:
    """
    Extracts linguistic features from essays for interpretability
    """
    
    def __init__(self):
        """Initialize the feature extractor"""
        self.stop_words = set(stopwords.words('english'))
    
    # ==================== SURFACE-LEVEL FEATURES ====================
    
    def extract_word_count(self, text: str) -> int:
        """
        Extract total word count
        
        Args:
            text (str): Essay text
            
        Returns:
            int: Total number of words
        """
        words = word_tokenize(text.lower())
        # Filter out punctuation
        words = [w for w in words if w.isalpha()]
        return len(words)
    
    def extract_sentence_count(self, text: str) -> int:
        """
        Extract total sentence count
        
        Args:
            text (str): Essay text
            
        Returns:
            int: Number of sentences
        """
        sentences = sent_tokenize(text)
        return len(sentences)
    
    def extract_avg_sentence_length(self, text: str) -> float:
        """
        Extract average sentence length (in words)
        
        Args:
            text (str): Essay text
            
        Returns:
            float: Average words per sentence
        """
        sentences = sent_tokenize(text)
        if len(sentences) == 0:
            return 0
        
        total_words = self.extract_word_count(text)
        return total_words / len(sentences)
    
    def extract_avg_word_length(self, text: str) -> float:
        """
        Extract average word length (in characters)
        
        Args:
            text (str): Essay text
            
        Returns:
            float: Average characters per word
        """
        words = word_tokenize(text.lower())
        words = [w for w in words if w.isalpha()]
        
        if len(words) == 0:
            return 0
        
        total_chars = sum(len(word) for word in words)
        return total_chars / len(words)
    
    # ==================== COMPLEXITY FEATURES ====================
    
    def extract_type_token_ratio(self, text: str) -> float:
        """
        Extract type-token ratio (vocabulary richness)
        Ratio of unique words to total words
        
        Args:
            text (str): Essay text
            
        Returns:
            float: Type-token ratio (0-1)
        """
        words = word_tokenize(text.lower())
        words = [w for w in words if w.isalpha()]
        
        if len(words) == 0:
            return 0
        
        unique_words = set(words)
        return len(unique_words) / len(words)
    
    def extract_complex_word_count(self, text: str) -> int:
        """
        Extract count of complex words (words with 3+ syllables)
        
        Args:
            text (str): Essay text
            
        Returns:
            int: Number of complex words
        """
        words = word_tokenize(text.lower())
        words = [w for w in words if w.isalpha()]
        
        complex_words = 0
        for word in words:
            syllables = textstat.syllable_count(word)
            if syllables >= 3:
                complex_words += 1
        
        return complex_words
    
    def extract_complex_word_ratio(self, text: str) -> float:
        """
        Extract ratio of complex words to total words
        
        Args:
            text (str): Essay text
            
        Returns:
            float: Complex word ratio
        """
        words = word_tokenize(text.lower())
        words = [w for w in words if w.isalpha()]
        
        if len(words) == 0:
            return 0
        
        complex_count = self.extract_complex_word_count(text)
        return complex_count / len(words)
    
    # ==================== READABILITY MEASURES ====================
    
    def extract_flesch_kincaid_grade(self, text: str) -> float:
        """
        Extract Flesch-Kincaid Grade Level
        Estimates US grade level needed to understand the text
        
        Args:
            text (str): Essay text
            
        Returns:
            float: Grade level
        """
        try:
            return textstat.flesch_kincaid_grade(text)
        except:
            return 0
    
    def extract_flesch_reading_ease(self, text: str) -> float:
        """
        Extract Flesch Reading Ease Score
        100-90: Very Easy, 90-80: Easy, 80-70: Fairly Easy, etc.
        
        Args:
            text (str): Essay text
            
        Returns:
            float: Reading ease score (0-100)
        """
        try:
            return textstat.flesch_reading_ease(text)
        except:
            return 50
    
    def extract_gunning_fog_index(self, text: str) -> float:
        """
        Extract Gunning Fog Index
        Years of education needed to understand text
        
        Args:
            text (str): Essay text
            
        Returns:
            float: Grade level equivalent
        """
        try:
            return textstat.gunning_fog(text)
        except:
            return 0
    
    def extract_smog_index(self, text: str) -> float:
        """
        Extract SMOG (Simple Measure of Gobbledygook) Index
        
        Args:
            text (str): Essay text
            
        Returns:
            float: Grade level
        """
        try:
            return textstat.smog_index(text)
        except:
            return 0
    
    def extract_dale_chall_score(self, text: str) -> float:
        """
        Extract Dale-Chall Readability Score
        Uses a list of 3000 easy words
        
        Args:
            text (str): Essay text
            
        Returns:
            float: Dale-Chall score
        """
        try:
            return textstat.dale_chall_readability_score(text)
        except:
            return 0
    
    # ==================== LEXICAL FEATURES ====================
    
    def extract_unique_word_count(self, text: str) -> int:
        """
        Extract number of unique words
        
        Args:
            text (str): Essay text
            
        Returns:
            int: Count of unique words
        """
        words = word_tokenize(text.lower())
        words = [w for w in words if w.isalpha()]
        return len(set(words))
    
    def extract_pronoun_count(self, text: str) -> int:
        """
        Extract count of pronouns (I, me, you, he, she, it, we, they)
        
        Args:
            text (str): Essay text
            
        Returns:
            int: Number of pronouns
        """
        pronouns = ['i', 'me', 'you', 'he', 'she', 'it', 'we', 'they', 'him', 'her', 'us', 'them']
        words = word_tokenize(text.lower())
        return sum(1 for word in words if word in pronouns)
    
    def extract_adverb_count(self, text: str) -> int:
        """
        Extract count of common adverbs (words ending in -ly)
        
        Args:
            text (str): Essay text
            
        Returns:
            int: Number of adverbs
        """
        words = word_tokenize(text.lower())
        return sum(1 for word in words if word.endswith('ly') and word.isalpha())
    
    # ==================== MAIN EXTRACTION METHOD ====================
    
    def extract_all_features(self, text: str) -> Dict[str, Union[int, float]]:
        """
        Extract all linguistic features from essay
        
        Args:
            text (str): Essay text
            
        Returns:
            Dict: Dictionary containing all extracted features
        """
        features = {
            'word_count': self.extract_word_count(text),
            'sentence_count': self.extract_sentence_count(text),
            'avg_sentence_length': self.extract_avg_sentence_length(text),
            'avg_word_length': self.extract_avg_word_length(text),
            'type_token_ratio': self.extract_type_token_ratio(text),
            'flesch_kincaid_grade': self.extract_flesch_kincaid_grade(text),
            'flesch_reading_ease': self.extract_flesch_reading_ease(text),
            'gunning_fog_index': self.extract_gunning_fog_index(text),
            'smog_index': self.extract_smog_index(text),
            'dale_chall_score': self.extract_dale_chall_score(text),
            'complex_word_count': self.extract_complex_word_count(text),
            'complex_word_ratio': self.extract_complex_word_ratio(text),
            'unique_word_count': self.extract_unique_word_count(text),
            'pronoun_count': self.extract_pronoun_count(text),
            'adverb_count': self.extract_adverb_count(text),
        }
        
        return features
    
    def extract_features_batch(self, texts: List[str]) -> pd.DataFrame:
        """
        Extract features for multiple essays
        
        Args:
            texts (List[str]): List of essay texts
            
        Returns:
            pd.DataFrame: DataFrame with features for each essay
        """
        features_list = []
        
        for i, text in enumerate(texts):
            features = self.extract_all_features(text)
            features_list.append(features)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Extracted features for {i + 1}/{len(texts)} essays")
        
        df_features = pd.DataFrame(features_list)
        
        logger.info(f"Feature extraction complete. Shape: {df_features.shape}")
        
        return df_features


# Example usage
if __name__ == "__main__":
    # Sample essay
    sample_essay = """
    The environment is very important for our future. We need to protect the trees and water.
    Many animals are losing their homes because of pollution. People should recycle more often.
    Governments should make stronger laws to protect nature. If we don't act now, future
    generations will suffer. Everyone has a responsibility to help the environment.
    """
    
    # Initialize extractor
    extractor = LinguisticFeatureExtractor()
    
    # Extract features
    features = extractor.extract_all_features(sample_essay)
    
    print("Extracted Features:")
    for feature, value in features.items():
        print(f"  {feature}: {value:.2f}")
