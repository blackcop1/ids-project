"""Utility functions for IDS project"""

import os
import json
import yaml
from pathlib import Path
from loguru import logger
import numpy as np
import pandas as pd
from datetime import datetime

class Logger:
    """Custom logger with file and console output"""
    
    def __init__(self, log_file='logs/ids.log'):
        """Initialize logger"""
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Remove default handler
        logger.remove()
        
        # Add console handler
        logger.add(
            lambda msg: print(msg, end=''),
            format='<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>'
        )
        
        # Add file handler
        logger.add(
            log_file,
            rotation='500 MB',
            retention='10 days',
            format='{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}'
        )
        
        self.logger = logger
    
    def info(self, message):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message):
        """Log error message"""
        self.logger.error(message)
    
    def debug(self, message):
        """Log debug message"""
        self.logger.debug(message)


class ConfigLoader:
    """Load configuration from YAML file"""
    
    @staticmethod
    def load_config(config_path='config/config.yaml'):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            print(f'Config file not found: {config_path}')
            return {}


class DataValidator:
    """Validate data integrity and quality"""
    
    @staticmethod
    def check_missing_values(df):
        """Check for missing values in dataframe"""
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print('Missing values detected:')
            print(missing[missing > 0])
            return False
        return True
    
    @staticmethod
    def check_data_types(df):
        """Verify data types"""
        return df.dtypes
    
    @staticmethod
    def get_dataset_statistics(df):
        """Get basic statistics about dataset"""
        stats = {
            'shape': df.shape,
            'missing_values': df.isnull().sum().sum(),
            'duplicates': df.duplicated().sum(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
            'numeric_columns': df.select_dtypes(include=[np.number]).shape[1],
            'categorical_columns': df.select_dtypes(include=['object']).shape[1]
        }
        return stats


class PathManager:
    """Manage project paths"""
    
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / 'data'
    MODELS_DIR = BASE_DIR / 'models'
    RESULTS_DIR = BASE_DIR / 'results'
    CONFIG_DIR = BASE_DIR / 'config'
    LOGS_DIR = BASE_DIR / 'logs'
    NOTEBOOKS_DIR = BASE_DIR / 'notebooks'
    
    @classmethod
    def create_dirs(cls):
        """Create all required directories"""
        for directory in [cls.DATA_DIR, cls.MODELS_DIR, cls.RESULTS_DIR, 
                         cls.CONFIG_DIR, cls.LOGS_DIR, cls.NOTEBOOKS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_model_path(cls, model_name):
        """Get path to model file"""
        return cls.MODELS_DIR / f'{model_name}.pkl'
    
    @classmethod
    def get_result_path(cls, result_name):
        """Get path to result file"""
        return cls.RESULTS_DIR / result_name


class TimeUtil:
    """Time utility functions"""
    
    @staticmethod
    def get_timestamp():
        """Get current timestamp"""
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    @staticmethod
    def get_date():
        """Get current date"""
        return datetime.now().strftime('%Y-%m-%d')


def print_separator(char='=', length=80):
    """Print separator line"""
    print(char * length)


def print_section(title, char='='):
    """Print section header"""
    print()
    print_separator(char, len(title) + 4)
    print(f' {title} ')
    print_separator(char, len(title) + 4)
    print()


def safe_divide(a, b, default=0):
    """Safe division to avoid division by zero"""
    return a / b if b != 0 else default
