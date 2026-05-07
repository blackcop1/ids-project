"""IDS Project - AI/ML-Based Intrusion Detection System"""

__version__ = '1.0.0'
__author__ = 'blackcop1'
__description__ = 'AI/ML-Based Intrusion Detection System for Network Traffic Analysis'

from .data_preprocessing import DataPreprocessor
from .feature_engineering import FeatureEngineer
from .model_training import ModelTrainer
from .model_evaluation import ModelEvaluator
from .model_persistence import ModelPersistence
from .utils import Logger

__all__ = [
    'DataPreprocessor',
    'FeatureEngineer',
    'ModelTrainer',
    'ModelEvaluator',
    'ModelPersistence',
    'Logger',
]
