"""Model training module for IDS project"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from xgboost import XGBClassifier
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from src.utils import Logger, print_section
import time


class ModelTrainer:
    """Train machine learning models for IDS"""
    
    def __init__(self):
        """Initialize model trainer"""
        self.logger = Logger()
        self.models = {}
        self.training_times = {}
    
    def train_random_forest(self, X_train, y_train, n_estimators=100, max_depth=20):
        """Train Random Forest classifier
        
        Args:
            X_train: Training features
            y_train: Training labels
            n_estimators: Number of trees
            max_depth: Maximum tree depth
        
        Returns:
            Trained Random Forest model
        """
        print_section('Training Random Forest Classifier')
        
        start_time = time.time()
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        self.logger.info(f'Training with {n_estimators} trees...')
        model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        self.training_times['Random Forest'] = training_time
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, n_jobs=-1)
        
        self.logger.info(f'Training time: {training_time:.2f}s')
        self.logger.info(f'Cross-validation scores: {cv_scores}')
        self.logger.info(f'Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})')
        
        self.models['Random Forest'] = model
        return model
    
    def train_xgboost(self, X_train, y_train, n_estimators=100, learning_rate=0.1):
        """Train XGBoost classifier
        
        Args:
            X_train: Training features
            y_train: Training labels
            n_estimators: Number of boosting rounds
            learning_rate: Learning rate
        
        Returns:
            Trained XGBoost model
        """
        print_section('Training XGBoost Classifier')
        
        start_time = time.time()
        
        # Determine number of classes for multi-class classification
        num_classes = len(np.unique(y_train))
        
        model = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            tree_method='hist',
            num_class=num_classes if num_classes > 2 else None,
            verbosity=1
        )
        
        self.logger.info(f'Training with {n_estimators} boosting rounds...')
        model.fit(X_train, y_train, eval_metric='mlogloss')
        
        training_time = time.time() - start_time
        self.training_times['XGBoost'] = training_time
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, n_jobs=-1)
        
        self.logger.info(f'Training time: {training_time:.2f}s')
        self.logger.info(f'Cross-validation scores: {cv_scores}')
        self.logger.info(f'Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})')
        
        self.models['XGBoost'] = model
        return model
    
    def train_neural_network(self, X_train, y_train, num_classes, epochs=20, batch_size=32):
        """Train Neural Network using TensorFlow/Keras
        
        Args:
            X_train: Training features
            y_train: Training labels
            num_classes: Number of output classes
            epochs: Number of epochs
            batch_size: Batch size
        
        Returns:
            Trained Neural Network model
        """
        print_section('Training Neural Network')
        
        input_dim = X_train.shape[1]
        
        model = Sequential([
            Dense(256, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            Dropout(0.2),
            
            Dense(num_classes, activation='softmax')
        ])
        
        optimizer = Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.logger.info(f'Model architecture:')
        model.summary()
        
        # Early stopping callback
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        start_time = time.time()
        
        self.logger.info(f'Training for {epochs} epochs...')
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=1
        )
        
        training_time = time.time() - start_time
        self.training_times['Neural Network'] = training_time
        
        self.logger.info(f'Training time: {training_time:.2f}s')
        self.logger.info(f'Final training accuracy: {history.history["accuracy"][-1]:.4f}')
        self.logger.info(f'Final validation accuracy: {history.history["val_accuracy"][-1]:.4f}')
        
        self.models['Neural Network'] = model
        return model, history
    
    def hyperparameter_tuning(self, X_train, y_train, model_type='random_forest'):
        """Perform hyperparameter tuning using GridSearchCV"""
        print_section('Hyperparameter Tuning')
        
        if model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
            base_model = RandomForestClassifier(random_state=42)
        
        elif model_type == 'xgboost':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3]
            }
            base_model = XGBClassifier(random_state=42)
        
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=3,
            n_jobs=-1,
            verbose=1
        )
        
        self.logger.info(f'Running grid search for {model_type}...')
        grid_search.fit(X_train, y_train)
        
        self.logger.info(f'Best parameters: {grid_search.best_params_}')
        self.logger.info(f'Best CV score: {grid_search.best_score_:.4f}')
        
        return grid_search.best_estimator_
    
    def train_all_models(self, X_train, y_train):
        """Train all three models"""
        print_section('Training All Models', '=')
        
        num_classes = len(np.unique(y_train))
        
        # Train Random Forest
        self.train_random_forest(X_train, y_train)
        
        # Train XGBoost
        self.train_xgboost(X_train, y_train)
        
        # Train Neural Network
        self.train_neural_network(X_train, y_train, num_classes)
        
        print_section('All Models Trained Successfully', '=')
        
        return self.models
    
    def get_models(self):
        """Get all trained models"""
        return self.models
    
    def get_training_times(self):
        """Get training times for all models"""
        return self.training_times
