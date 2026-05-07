"""Model evaluation module for IDS project"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score, precision_recall_curve
)
from src.utils import Logger, print_section
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Evaluate and compare model performance"""
    
    def __init__(self, label_encoder=None):
        """Initialize model evaluator
        
        Args:
            label_encoder: LabelEncoder for decoding labels
        """
        self.logger = Logger()
        self.label_encoder = label_encoder
        self.results = {}
        
        # Set plotting style
        sns.set_style('darkgrid')
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def evaluate_model(self, model, X_test, y_test, model_name='Model'):
        """Evaluate a single model
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
        
        Returns:
            Dictionary with evaluation metrics
        """
        print_section(f'Evaluating {model_name}')
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # For neural networks, get class predictions
        if hasattr(y_pred, 'shape') and len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        results = {
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Predictions': y_pred
        }
        
        self.logger.info(f'Accuracy: {accuracy:.4f}')
        self.logger.info(f'Precision: {precision:.4f}')
        self.logger.info(f'Recall: {recall:.4f}')
        self.logger.info(f'F1-Score: {f1:.4f}')
        
        self.results[model_name] = results
        return results
    
    def print_classification_report(self, y_test, y_pred, model_name='Model'):
        """Print detailed classification report"""
        print_section(f'Classification Report - {model_name}')
        
        if self.label_encoder:
            target_names = self.label_encoder.classes_
        else:
            target_names = [str(i) for i in np.unique(y_test)]
        
        report = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
        print(report)
        self.logger.info(f'Classification report printed')
    
    def plot_confusion_matrix(self, y_test, y_pred, model_name='Model'):
        """Plot confusion matrix"""
        print_section(f'Plotting Confusion Matrix - {model_name}')
        
        cm = confusion_matrix(y_test, y_pred)
        
        if self.label_encoder:
            labels = self.label_encoder.classes_
        else:
            labels = np.unique(y_test)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels, cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f'results/confusion_matrix_{model_name}.png', dpi=300, bbox_inches='tight')
        self.logger.info(f'Confusion matrix saved to results/confusion_matrix_{model_name}.png')
        plt.close()
    
    def plot_roc_curve(self, y_test, y_pred_proba, model_name='Model'):
        """Plot ROC curve (for binary classification)"""
        print_section(f'Plotting ROC Curve - {model_name}')
        
        num_classes = len(np.unique(y_test))
        
        plt.figure(figsize=(10, 8))
        
        if num_classes == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        else:
            # Multi-class classification - One-vs-Rest
            from sklearn.preprocessing import label_binarize
            y_test_bin = label_binarize(y_test, classes=np.arange(num_classes))
            
            for i in range(num_classes):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f'results/roc_curve_{model_name}.png', dpi=300, bbox_inches='tight')
        self.logger.info(f'ROC curve saved to results/roc_curve_{model_name}.png')
        plt.close()
    
    def plot_feature_importance(self, model, feature_names, model_name='Model'):
        """Plot feature importance"""
        print_section(f'Plotting Feature Importance - {model_name}')
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            self.logger.warning(f'{model_name} does not have feature importance')
            return
        
        # Get top 20 features
        indices = np.argsort(importances)[-20:][::-1]
        
        plt.figure(figsize=(10, 8))
        plt.title(f'Top 20 Feature Importances - {model_name}')
        plt.bar(range(len(indices)), importances[indices], align='center')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f'results/feature_importance_{model_name}.png', dpi=300, bbox_inches='tight')
        self.logger.info(f'Feature importance plot saved')
        plt.close()
    
    def compare_models(self):
        """Compare performance of all models"""
        print_section('Model Comparison', '=')
        
        comparison_df = pd.DataFrame([
            {
                'Model': results['Model'],
                'Accuracy': results['Accuracy'],
                'Precision': results['Precision'],
                'Recall': results['Recall'],
                'F1-Score': results['F1-Score']
            }
            for results in self.results.values()
        ])
        
        self.logger.info('\nModel Performance Comparison:')
        self.logger.info(str(comparison_df.to_string(index=False)))
        
        # Save comparison
        comparison_df.to_csv('results/model_comparison.csv', index=False)
        self.logger.info('Comparison saved to results/model_comparison.csv')
        
        return comparison_df
    
    def plot_model_comparison(self):
        """Plot model comparison bar chart"""
        print_section('Plotting Model Comparison')
        
        comparison_df = pd.DataFrame([
            {
                'Model': results['Model'],
                'Accuracy': results['Accuracy'],
                'Precision': results['Precision'],
                'Recall': results['Recall'],
                'F1-Score': results['F1-Score']
            }
            for results in self.results.values()
        ])
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            comparison_df.plot(x='Model', y=metric, kind='bar', ax=ax, legend=False)
            ax.set_title(f'{metric} Comparison')
            ax.set_ylabel(metric)
            ax.set_xlabel('')
            ax.set_ylim([0, 1.05])
            ax.grid(axis='y', alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
        self.logger.info('Model comparison plot saved to results/model_comparison.png')
        plt.close()
