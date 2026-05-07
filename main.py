"""Main execution script for IDS project"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer
from src.model_evaluation import ModelEvaluator
from src.model_persistence import ModelPersistence
from src.utils import Logger, PathManager, print_section
import numpy as np

def main():
    """Main execution pipeline"""
    
    # Initialize
    logger = Logger()
    PathManager.create_dirs()
    
    print_section('IDS Project - Complete Pipeline', '=')
    
    # Check if dataset exists
    dataset_path = 'data/UNSW-NB15_training-set.csv'
    if not os.path.exists(dataset_path):
        logger.error(f'Dataset not found: {dataset_path}')
        logger.info('Please download the dataset first:')
        logger.info('  python data/download_dataset.py')
        return
    
    # Step 1: Data Preprocessing
    logger.info('Step 1: Data Preprocessing')
    preprocessor = DataPreprocessor(dataset_path)
    X_train, X_test, y_train, y_test = preprocessor.prepare_data()
    
    num_classes = len(np.unique(y_train))
    logger.info(f'Number of classes: {num_classes}')
    
    # Step 2: Model Training
    logger.info('\nStep 2: Model Training')
    trainer = ModelTrainer()
    
    rf_model = trainer.train_random_forest(X_train, y_train)
    xgb_model = trainer.train_xgboost(X_train, y_train)
    nn_model, nn_history = trainer.train_neural_network(X_train, y_train, num_classes)
    
    # Step 3: Model Evaluation
    logger.info('\nStep 3: Model Evaluation')
    evaluator = ModelEvaluator(label_encoder=preprocessor.label_encoder)
    
    # Evaluate Random Forest
    rf_results = evaluator.evaluate_model(rf_model, X_test, y_test, 'Random Forest')
    evaluator.print_classification_report(y_test, rf_results['Predictions'], 'Random Forest')
    evaluator.plot_confusion_matrix(y_test, rf_results['Predictions'], 'Random Forest')
    evaluator.plot_roc_curve(y_test, rf_model.predict_proba(X_test), 'Random Forest')
    evaluator.plot_feature_importance(rf_model, preprocessor.feature_columns, 'Random Forest')
    
    # Evaluate XGBoost
    xgb_results = evaluator.evaluate_model(xgb_model, X_test, y_test, 'XGBoost')
    evaluator.print_classification_report(y_test, xgb_results['Predictions'], 'XGBoost')
    evaluator.plot_confusion_matrix(y_test, xgb_results['Predictions'], 'XGBoost')
    evaluator.plot_roc_curve(y_test, xgb_model.predict_proba(X_test), 'XGBoost')
    evaluator.plot_feature_importance(xgb_model, preprocessor.feature_columns, 'XGBoost')
    
    # Evaluate Neural Network
    nn_predictions = nn_model.predict(X_test, verbose=0)
    nn_pred_classes = np.argmax(nn_predictions, axis=1)
    nn_results = {
        'Model': 'Neural Network',
        'Predictions': nn_pred_classes
    }
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    nn_results['Accuracy'] = accuracy_score(y_test, nn_pred_classes)
    nn_results['Precision'] = precision_score(y_test, nn_pred_classes, average='weighted', zero_division=0)
    nn_results['Recall'] = recall_score(y_test, nn_pred_classes, average='weighted', zero_division=0)
    nn_results['F1-Score'] = f1_score(y_test, nn_pred_classes, average='weighted', zero_division=0)
    evaluator.results['Neural Network'] = nn_results
    
    evaluator.print_classification_report(y_test, nn_pred_classes, 'Neural Network')
    evaluator.plot_confusion_matrix(y_test, nn_pred_classes, 'Neural Network')
    evaluator.plot_roc_curve(y_test, nn_predictions, 'Neural Network')
    
    # Step 4: Model Comparison
    logger.info('\nStep 4: Model Comparison')
    comparison_df = evaluator.compare_models()
    evaluator.plot_model_comparison()
    
    # Step 5: Save Models
    logger.info('\nStep 5: Saving Models')
    persistence = ModelPersistence()
    persistence.save_model(rf_model, 'random_forest_model')
    persistence.save_model(xgb_model, 'xgboost_model')
    persistence.save_model(nn_model, 'neural_network_model')
    persistence.save_preprocessor(
        preprocessor.scaler,
        preprocessor.label_encoder,
        preprocessor.feature_columns
    )
    
    print_section('Pipeline Completed Successfully', '=')
    logger.info('All results saved to results/ directory')
    logger.info('All models saved to models/ directory')
    logger.info('\nNext steps:')
    logger.info('1. Review results in results/ directory')
    logger.info('2. Use trained models for real-time detection')
    logger.info('3. Customize thresholds in config/config.yaml')
    logger.info('4. Run real-time detection: python src/real_time_detection.py')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n\nPipeline interrupted by user')
    except Exception as e:
        print(f'\nError: {str(e)}')
        import traceback
        traceback.print_exc()
