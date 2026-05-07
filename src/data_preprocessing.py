"""Data preprocessing module for IDS project"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from src.utils import Logger, DataValidator, print_section
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Preprocess network traffic data for IDS"""
    
    def __init__(self, dataset_path, random_state=42, test_size=0.2):
        """Initialize preprocessor
        
        Args:
            dataset_path: Path to CSV dataset
            random_state: Random seed for reproducibility
            test_size: Test set size (0-1)
        """
        self.dataset_path = dataset_path
        self.random_state = random_state
        self.test_size = test_size
        self.logger = Logger()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.target_column = None
        self.df = None
        
    def load_data(self):
        """Load dataset from CSV file"""
        print_section('Loading Dataset')
        try:
            self.df = pd.read_csv(self.dataset_path)
            self.logger.info(f'Dataset loaded successfully: {self.df.shape}')
            self.logger.info(f'Columns: {list(self.df.columns)}')
            return self.df
        except FileNotFoundError:
            self.logger.error(f'Dataset not found: {self.dataset_path}')
            raise
        except Exception as e:
            self.logger.error(f'Error loading dataset: {str(e)}')
            raise
    
    def explore_data(self):
        """Explore and display data statistics"""
        print_section('Data Exploration')
        
        stats = DataValidator.get_dataset_statistics(self.df)
        for key, value in stats.items():
            self.logger.info(f'{key}: {value}')
        
        # Check for missing values
        if DataValidator.check_missing_values(self.df):
            self.logger.info('No missing values detected')
        
        # Display class distribution
        if 'Label' in self.df.columns:
            print_section('Class Distribution')
            class_dist = self.df['Label'].value_counts()
            for class_name, count in class_dist.items():
                percentage = (count / len(self.df)) * 100
                self.logger.info(f'{class_name}: {count} ({percentage:.2f}%)')
    
    def handle_missing_values(self, strategy='drop'):
        """Handle missing values
        
        Args:
            strategy: 'drop' or 'mean' or 'median'
        """
        print_section('Handling Missing Values')
        
        missing_before = self.df.isnull().sum().sum()
        
        if strategy == 'drop':
            self.df = self.df.dropna()
        elif strategy == 'mean':
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
        elif strategy == 'median':
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        
        missing_after = self.df.isnull().sum().sum()
        self.logger.info(f'Missing values before: {missing_before}, after: {missing_after}')
    
    def remove_duplicates(self):
        """Remove duplicate rows"""
        print_section('Removing Duplicates')
        
        duplicates_before = self.df.duplicated().sum()
        self.df = self.df.drop_duplicates()
        duplicates_after = self.df.duplicated().sum()
        
        self.logger.info(f'Duplicates removed: {duplicates_before - duplicates_after}')
        self.logger.info(f'Dataset shape after removing duplicates: {self.df.shape}')
    
    def handle_categorical_features(self):
        """Encode categorical features"""
        print_section('Handling Categorical Features')
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col != 'Label':  # Don't encode label yet
                unique_values = self.df[col].nunique()
                self.logger.info(f'Encoding {col} ({unique_values} unique values)')
                self.df[col] = pd.factorize(self.df[col])[0]
    
    def remove_outliers(self, method='iqr', threshold=3):
        """Remove outliers from numeric columns
        
        Args:
            method: 'iqr' (Interquartile Range) or 'zscore'
            threshold: Z-score threshold for removal
        """
        print_section('Removing Outliers')
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        rows_before = len(self.df)
        
        if method == 'iqr':
            for col in numeric_cols:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        
        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(self.df[numeric_cols]))
            self.df = self.df[(z_scores < threshold).all(axis=1)]
        
        rows_after = len(self.df)
        self.logger.info(f'Outliers removed: {rows_before - rows_after}')
        self.logger.info(f'Dataset shape after removing outliers: {self.df.shape}')
    
    def prepare_features_and_labels(self):
        """Prepare features and labels for training"""
        print_section('Preparing Features and Labels')
        
        # Identify target column
        if 'Label' in self.df.columns:
            self.target_column = 'Label'
        elif 'label' in self.df.columns:
            self.target_column = 'label'
        else:
            self.logger.error('Label column not found')
            raise ValueError('Label column required')
        
        # Separate features and labels
        self.feature_columns = [col for col in self.df.columns if col != self.target_column]
        X = self.df[self.feature_columns]
        y = self.df[self.target_column]
        
        self.logger.info(f'Features: {len(self.feature_columns)}')
        self.logger.info(f'Samples: {len(X)}')
        
        return X, y
    
    def encode_labels(self, y):
        """Encode target labels"""
        print_section('Encoding Labels')
        
        y_encoded = self.label_encoder.fit_transform(y)
        
        label_mapping = dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))
        self.logger.info('Label mapping:')
        for label, code in label_mapping.items():
            self.logger.info(f'  {label}: {code}')
        
        return y_encoded
    
    def normalize_features(self, X):
        """Normalize features using StandardScaler"""
        print_section('Normalizing Features')
        
        X_scaled = self.scaler.fit_transform(X)
        self.logger.info(f'Features normalized using StandardScaler')
        self.logger.info(f'Mean after scaling: {X_scaled.mean():.6f}')
        self.logger.info(f'Std after scaling: {X_scaled.std():.6f}')
        
        return X_scaled
    
    def split_data(self, X, y):
        """Split data into training and testing sets"""
        print_section('Splitting Data')
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        self.logger.info(f'Training set size: {len(X_train)} ({(len(X_train)/len(X))*100:.2f}%)')
        self.logger.info(f'Testing set size: {len(X_test)} ({(len(X_test)/len(X))*100:.2f}%)')
        
        return X_train, X_test, y_train, y_test
    
    def prepare_data(self, remove_outliers=True):
        """Complete data preparation pipeline
        
        Returns:
            X_train, X_test, y_train, y_test (numpy arrays)
        """
        print_section('Starting Data Preparation Pipeline', '=')
        
        # Load data
        self.load_data()
        
        # Explore data
        self.explore_data()
        
        # Preprocessing steps
        self.handle_missing_values(strategy='drop')
        self.remove_duplicates()
        self.handle_categorical_features()
        
        if remove_outliers:
            self.remove_outliers(method='iqr')
        
        # Prepare features and labels
        X, y = self.prepare_features_and_labels()
        
        # Encode labels
        y_encoded = self.encode_labels(y)
        
        # Normalize features
        X_scaled = self.normalize_features(X)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X_scaled, y_encoded)
        
        print_section('Data Preparation Complete', '=')
        
        return X_train, X_test, y_train, y_test
    
    def get_preprocessor_objects(self):
        """Get scaler and label encoder for later use"""
        return {
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns
        }
