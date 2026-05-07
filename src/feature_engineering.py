"""Feature engineering module for IDS project"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from src.utils import Logger, print_section


class FeatureEngineer:
    """Engineer and select features for IDS"""
    
    def __init__(self):
        """Initialize feature engineer"""
        self.logger = Logger()
    
    def extract_statistical_features(self, data):
        """Extract statistical features from network traffic"""
        print_section('Extracting Statistical Features')
        
        features = {}
        
        # Basic statistics
        for col in data.select_dtypes(include=[np.number]).columns:
            features[f'{col}_mean'] = data[col].mean()
            features[f'{col}_std'] = data[col].std()
            features[f'{col}_min'] = data[col].min()
            features[f'{col}_max'] = data[col].max()
            features[f'{col}_median'] = data[col].median()
            features[f'{col}_skew'] = data[col].skew()
        
        self.logger.info(f'Extracted {len(features)} statistical features')
        return pd.DataFrame([features])
    
    def create_polynomial_features(self, X, degree=2, include_bias=False):
        """Create polynomial features
        
        Args:
            X: Input features (numpy array or dataframe)
            degree: Polynomial degree
            include_bias: Whether to include bias term
        
        Returns:
            Polynomial features array
        """
        print_section('Creating Polynomial Features')
        
        poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
        X_poly = poly.fit_transform(X)
        
        self.logger.info(f'Original features: {X.shape[1]}')
        self.logger.info(f'Polynomial features: {X_poly.shape[1]}')
        
        return X_poly
    
    def get_feature_importance_names(self, feature_names, degree=2):
        """Get names of polynomial features"""
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly.fit(np.zeros((1, len(feature_names))))
        return poly.get_feature_names_out(feature_names)
    
    def create_interaction_features(self, X, feature_names):
        """Create interaction features between important feature pairs"""
        print_section('Creating Interaction Features')
        
        X_interactions = X.copy()
        interaction_names = []
        
        # Create some key interactions for network traffic
        # This would depend on domain knowledge about what features interact
        
        self.logger.info(f'Created {len(interaction_names)} interaction features')
        return X_interactions, interaction_names
    
    @staticmethod
    def select_important_features(X, y, model, top_k=20):
        """Select top-k important features based on model
        
        Args:
            X: Training features
            y: Training labels
            model: Trained model with feature_importances_
            top_k: Number of top features to select
        
        Returns:
            Indices of top-k features
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[-top_k:][::-1]
            return indices
        else:
            # For models without feature_importances_, use coefficients
            if hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0])
                indices = np.argsort(importances)[-top_k:][::-1]
                return indices
        
        return np.arange(min(top_k, X.shape[1]))
    
    @staticmethod
    def scale_features(X, scaler):
        """Scale features using provided scaler"""
        return scaler.transform(X)
