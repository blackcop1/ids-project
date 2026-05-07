from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='ids-project',
    version='1.0.0',
    author='blackcop1',
    description='AI/ML-Based Intrusion Detection System for Network Traffic Analysis',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/blackcop1/ids-project',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'Topic :: Security',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.24.3',
        'pandas>=2.0.3',
        'scikit-learn>=1.3.0',
        'xgboost>=2.0.0',
        'tensorflow>=2.13.0',
        'matplotlib>=3.7.2',
        'seaborn>=0.12.2',
        'plotly>=5.16.1',
        'scapy>=2.5.0',
        'joblib>=1.3.1',
        'pyyaml>=6.0.1',
        'tqdm>=4.66.1',
        'loguru>=0.7.0',
    ],
)
