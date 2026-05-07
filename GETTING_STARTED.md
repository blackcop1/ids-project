# IDS Project - Getting Started

## Quick Start Guide

### 1. Clone Repository
```bash
git clone https://github.com/blackcop1/ids-project.git
cd ids-project
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Dataset
```bash
python data/download_dataset.py
```

### 5. Run Complete Pipeline
```bash
python main.py
```

## Project Structure

```
ids-project/
├── src/                          # Source code
│   ├── data_preprocessing.py     # Data loading and cleaning
│   ├── feature_engineering.py    # Feature extraction
│   ├── model_training.py         # Model training
│   ├── model_evaluation.py       # Evaluation and visualization
│   ├── real_time_detection.py    # Real-time detection
│   ├── model_persistence.py      # Save/load models
│   └── utils.py                  # Utility functions
├── data/                         # Datasets
├── models/                       # Trained models
├── results/                      # Evaluation results
├── config/                       # Configuration files
└── notebooks/                    # Jupyter notebooks
```

## Usage Examples

### Data Preprocessing
```python
from src.data_preprocessing import DataPreprocessor

preprocessor = DataPreprocessor('data/UNSW-NB15_training-set.csv')
X_train, X_test, y_train, y_test = preprocessor.prepare_data()
```

### Model Training
```python
from src.model_training import ModelTrainer

trainer = ModelTrainer()
rf_model = trainer.train_random_forest(X_train, y_train)
xgb_model = trainer.train_xgboost(X_train, y_train)
nn_model, history = trainer.train_neural_network(X_train, y_train, num_classes=10)
```

### Model Evaluation
```python
from src.model_evaluation import ModelEvaluator

evaluator = ModelEvaluator(label_encoder=preprocessor.label_encoder)
evaluator.evaluate_model(rf_model, X_test, y_test, 'Random Forest')
evaluator.plot_confusion_matrix(y_test, rf_model.predict(X_test), 'Random Forest')
evaluator.plot_roc_curve(y_test, rf_model.predict_proba(X_test), 'Random Forest')
evaluator.compare_models()
```

### Save/Load Models
```python
from src.model_persistence import ModelPersistence

persistence = ModelPersistence()
persistence.save_model(rf_model, 'random_forest_model')
persistence.save_preprocessor(
    preprocessor.scaler,
    preprocessor.label_encoder,
    preprocessor.feature_columns
)

# Load later
loaded_model = persistence.load_model('random_forest_model')
scaler, le, features = persistence.load_preprocessor()
```

### Real-Time Detection
```bash
# Requires root/admin privileges
sudo python -c "
from src.real_time_detection import PacketSniffer, RealtimeDetector
from src.model_persistence import ModelPersistence

persistence = ModelPersistence()
model = persistence.load_model('random_forest_model')
scaler, le, features = persistence.load_preprocessor()

detector = RealtimeDetector(model, scaler, le, features)
sniffer = PacketSniffer(interface='eth0')

# Sniff and analyze packets
for _ in range(100):
    sniffer.start_sniffing(1)
    if sniffer.packets:
        packet = sniffer.packets[0]
        detector.process_packet(packet)
        sniffer.packets = []

detector.print_statistics()
"
```

## Jupyter Notebooks

For interactive analysis, use the provided Jupyter notebooks:

```bash
jupyter notebook notebooks/EDA_and_Model_Comparison.ipynb
```

## Configuration

Edit `config/config.yaml` to customize:
- Dataset paths
- Model hyperparameters
- Detection thresholds
- Network interfaces

## Troubleshooting

### Issue: "Permission denied" for real-time detection
```bash
# Use sudo
sudo python src/real_time_detection.py
```

### Issue: "Module not found" errors
```bash
# Ensure virtual environment is activated
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: Out of memory during training
```bash
# Use a smaller dataset sample
python src/model_training.py --sample 100000
```

## Performance Tips

1. **Use XGBoost for best performance** - Highest accuracy with reasonable training time
2. **Adjust batch size** - Increase for faster training, decrease to reduce memory
3. **Use feature selection** - Select top 20-30 features to speed up training
4. **Enable GPU acceleration** - TensorFlow can use GPU for neural networks
5. **Parallelize training** - Use `n_jobs=-1` for Random Forest and XGBoost

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to branch
5. Open a Pull Request

## License

MIT License - see LICENSE file

## References

- [UNSW-NB15 Dataset](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [TensorFlow/Keras](https://www.tensorflow.org/)
- [Scapy Documentation](https://scapy.readthedocs.io/)

## Support

For issues and questions:
1. Check existing GitHub issues
2. Create a new issue with detailed description
3. Include error messages and code snippets

---

**Happy Learning! Good luck with your cybersecurity journey!** 🚀
