"""Model persistence module for saving and loading models"""

import joblib
import os
from pathlib import Path
from src.utils import Logger, print_section


class ModelPersistence:
    """Save and load trained models and preprocessors"""
    
    def __init__(self, models_dir='models'):
        """Initialize model persistence
        
        Args:
            models_dir: Directory to store models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logger = Logger()
    
    def save_model(self, model, model_name):
        """Save a trained model to disk
        
        Args:
            model: Trained model object
            model_name: Name to save model as
        """
        print_section(f'Saving {model_name}')
        
        try:
            model_path = self.models_dir / f'{model_name}.pkl'
            
            # For neural networks, use different format
            if hasattr(model, 'save'):
                model_path = self.models_dir / f'{model_name}.h5'
                model.save(str(model_path))
            else:
                joblib.dump(model, str(model_path))
            
            self.logger.info(f'Model saved to {model_path}')
            return model_path
        except Exception as e:
            self.logger.error(f'Error saving model: {str(e)}')
            raise
    
    def load_model(self, model_name):
        """Load a trained model from disk
        
        Args:
            model_name: Name of model to load
        
        Returns:
            Loaded model
        """
        print_section(f'Loading {model_name}')
        
        try:
            # Try pickle format first
            model_path = self.models_dir / f'{model_name}.pkl'
            if model_path.exists():
                model = joblib.load(str(model_path))
            else:
                # Try HDF5 format for neural networks
                model_path = self.models_dir / f'{model_name}.h5'
                if model_path.exists():
                    from tensorflow.keras.models import load_model
                    model = load_model(str(model_path))
                else:
                    raise FileNotFoundError(f'Model not found: {model_name}')
            
            self.logger.info(f'Model loaded from {model_path}')
            return model
        except Exception as e:
            self.logger.error(f'Error loading model: {str(e)}')
            raise
    
    def save_preprocessor(self, scaler, label_encoder, feature_columns):
        """Save preprocessor objects (scaler, label encoder, feature columns)
        
        Args:
            scaler: StandardScaler object
            label_encoder: LabelEncoder object
            feature_columns: List of feature column names
        """
        print_section('Saving Preprocessors')
        
        try:
            scaler_path = self.models_dir / 'scaler.pkl'
            encoder_path = self.models_dir / 'label_encoder.pkl'
            features_path = self.models_dir / 'feature_columns.pkl'
            
            joblib.dump(scaler, str(scaler_path))
            joblib.dump(label_encoder, str(encoder_path))
            joblib.dump(feature_columns, str(features_path))
            
            self.logger.info(f'Scaler saved to {scaler_path}')
            self.logger.info(f'Label encoder saved to {encoder_path}')
            self.logger.info(f'Feature columns saved to {features_path}')
        except Exception as e:
            self.logger.error(f'Error saving preprocessors: {str(e)}')
            raise
    
    def load_preprocessor(self):
        """Load preprocessor objects
        
        Returns:
            Tuple of (scaler, label_encoder, feature_columns)
        """
        print_section('Loading Preprocessors')
        
        try:
            scaler_path = self.models_dir / 'scaler.pkl'
            encoder_path = self.models_dir / 'label_encoder.pkl'
            features_path = self.models_dir / 'feature_columns.pkl'
            
            scaler = joblib.load(str(scaler_path))
            label_encoder = joblib.load(str(encoder_path))
            feature_columns = joblib.load(str(features_path))
            
            self.logger.info(f'Preprocessors loaded successfully')
            return scaler, label_encoder, feature_columns
        except Exception as e:
            self.logger.error(f'Error loading preprocessors: {str(e)}')
            raise
    
    def list_saved_models(self):
        """List all saved models"""
        print_section('Saved Models')
        
        pkl_files = list(self.models_dir.glob('*.pkl'))
        h5_files = list(self.models_dir.glob('*.h5'))
        
        all_files = pkl_files + h5_files
        
        if not all_files:
            self.logger.info('No saved models found')
            return []
        
        for file_path in all_files:
            file_size = file_path.stat().st_size / (1024 * 1024)  # Convert to MB
            self.logger.info(f'{file_path.name} ({file_size:.2f} MB)')
        
        return all_files
    
    def delete_model(self, model_name):
        """Delete a saved model
        
        Args:
            model_name: Name of model to delete
        """
        try:
            model_path = self.models_dir / f'{model_name}.pkl'
            if model_path.exists():
                os.remove(model_path)
                self.logger.info(f'Model deleted: {model_path}')
            else:
                model_path = self.models_dir / f'{model_name}.h5'
                if model_path.exists():
                    os.remove(model_path)
                    self.logger.info(f'Model deleted: {model_path}')
                else:
                    self.logger.warning(f'Model not found: {model_name}')
        except Exception as e:
            self.logger.error(f'Error deleting model: {str(e)}')
            raise
