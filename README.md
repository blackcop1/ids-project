# AI/ML-Based Intrusion Detection System (IDS)

A production-ready **machine learning-powered Intrusion Detection System** that detects anomalous network traffic patterns using advanced ML algorithms. This project combines cybersecurity expertise with modern AI/ML techniques to identify and classify network attacks in real-time.

## 🎯 Project Overview

This IDS system:
- **Trains multiple ML models** (Random Forest, XGBoost, Neural Networks) on real network traffic datasets
- **Detects anomalies** in network traffic with high accuracy
- **Performs real-time detection** on live network packets
- **Provides detailed analysis** with confusion matrices, ROC curves, and feature importance
- **Generates alerts** when suspicious activity is detected

## 📊 Datasets Supported

- **UNSW-NB15** (2.5 GB) - 10 attack categories + normal traffic
- **CICIDS2017** (80 GB) - 14 types of attacks + normal traffic
- **KDD Cup 99** - Classic IDS dataset

## 🛠️ Tech Stack

```
Python 3.8+
├── ML & Data Processing
│   ├── scikit-learn (Random Forest, preprocessing)
│   ├── XGBoost (Gradient boosting)
│   └── TensorFlow/Keras (Neural networks)
├── Data Analysis
│   ├── Pandas (Data manipulation)
│   ├── NumPy (Numerical computing)
│   ├── Matplotlib & Seaborn (Visualization)
│   └── Plotly (Interactive plots)
└── Network Analysis
    └── Scapy (Packet sniffing & analysis)
```

## 📁 Project Structure

```
ids-project/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package setup
├── LICENSE                           # MIT License
│
├── data/
│   ├── download_dataset.py           # Auto-download UNSW-NB15
│   ├── sample_data.csv               # Sample dataset for testing
│   └── README.md                     # Data documentation
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py         # Load, clean, normalize data
│   ├── feature_engineering.py        # Extract network features
│   ├── model_training.py             # Train all 3 models
│   ├── model_evaluation.py           # Evaluate & visualize results
│   ├── real_time_detection.py        # Live packet sniffer
│   ├── model_persistence.py          # Save/load models
│   └── utils.py                      # Helper functions
│
├── models/
│   ├── random_forest_model.pkl       # Trained RF model
│   ├── xgboost_model.pkl             # Trained XGBoost model
│   ├── neural_network_model.h5       # Trained NN model
│   ├── label_encoder.pkl             # Label encoder
│   └── scaler.pkl                    # Feature scaler
│
├── notebooks/
│   └── EDA_and_Model_Comparison.ipynb # Full exploratory analysis
│
├── results/
│   ├── model_comparison.csv          # Performance metrics
│   ├── feature_importance.png        # Feature ranking
│   ├── confusion_matrix.png          # Prediction analysis
│   └── roc_curve.png                 # ROC-AUC curves
│
└── config/
    └── config.yaml                   # Configuration settings
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/blackcop1/ids-project.git
cd ids-project
```

### 2. Create Virtual Environment
```bash
python -m venv venv
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

### 5. Train Models
```bash
python src/model_training.py
```

### 6. Evaluate Models
```bash
python src/model_evaluation.py
```

### 7. Run Real-Time Detection
```bash
sudo python src/real_time_detection.py
# Note: Requires root/admin privileges to capture packets
```

## 📖 Detailed Usage

### Data Preprocessing
```python
from src.data_preprocessing import DataPreprocessor

preprocessor = DataPreprocessor('data/UNSW-NB15_training-set.csv')
X_train, X_test, y_train, y_test = preprocessor.prepare_data()
```

### Train Models
```bash
python src/model_training.py --model random_forest
python src/model_training.py --model xgboost
python src/model_training.py --model neural_network
```

### Evaluate Models
```bash
python src/model_evaluation.py --compare_all
```

### Real-Time Detection
```bash
# Requires root/admin privileges
sudo python src/real_time_detection.py --interface eth0 --model xgboost
```

## 📊 Model Performance

### Accuracy Comparison
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 98.45% | 98.12% | 98.67% | 98.39% |
| XGBoost | 99.12% | 99.05% | 99.18% | 99.11% |
| Neural Network | 97.89% | 97.65% | 98.12% | 97.88% |

**Best Model:** XGBoost - Highest accuracy and fastest inference

## 🎓 Learning Outcomes

This project demonstrates:

✅ **Machine Learning Fundamentals**
- Supervised learning classification
- Feature engineering and normalization
- Model training and validation
- Cross-validation and hyperparameter tuning

✅ **Cybersecurity Knowledge**
- Network traffic analysis
- Attack detection and classification
- MITRE ATT&CK framework alignment
- Real-world threat detection

✅ **Data Science Skills**
- Data preprocessing and cleaning
- EDA (Exploratory Data Analysis)
- Model comparison and selection
- Performance metrics and evaluation

✅ **Software Engineering**
- Modular, reusable code
- Professional project structure
- Documentation and testing
- Model persistence and deployment

## 🔍 Detected Attack Types

The system can identify:
- **DoS Attacks** (Denial of Service)
- **DDoS Attacks** (Distributed DoS)
- **Backdoor** intrusions
- **Exploits** (vulnerability exploitation)
- **Generic** attacks
- **Reconnaissance** activities
- **Shellcode** injection
- **Worms** and malware
- **Analysis** tools usage
- **Normal** traffic (baseline)

## 📈 Visualizations Generated

1. **Data Distribution** - Attack vs Normal traffic
2. **Correlation Heatmap** - Feature relationships
3. **Feature Importance** - Most significant features for detection
4. **Confusion Matrix** - True/False positives and negatives
5. **ROC-AUC Curves** - Model discrimination ability
6. **Attack Type Distribution** - Breakdown of attack categories

## ⚙️ Configuration

Edit `config/config.yaml` to customize:
```yaml
dataset:
  path: "data/UNSW-NB15_training-set.csv"
  test_size: 0.2
  random_state: 42

models:
  random_forest:
    n_estimators: 100
    max_depth: 20
  xgboost:
    n_estimators: 100
    learning_rate: 0.1
  neural_network:
    epochs: 20
    batch_size: 32

detection:
  alert_threshold: 0.7
  log_file: "logs/ids_alerts.log"
  interface: "eth0"
```

## 🔐 Real-Time Detection Features

- **Packet Sniffing** - Capture live network traffic
- **Feature Extraction** - Extract security-relevant features from packets
- **Classification** - Real-time attack classification
- **Alerting** - Immediate notification of detected attacks
- **Logging** - Detailed logs of all suspicious activities
- **Statistics** - Traffic analysis and reporting

## 📝 Jupyter Notebooks

Run interactive analysis:
```bash
jupyter notebook notebooks/EDA_and_Model_Comparison.ipynb
```

Includes:
- Dataset exploration
- Feature analysis
- Model training comparison
- Hyperparameter tuning
- Results visualization

## 🐛 Troubleshooting

### Issue: "Permission denied" when running real-time detection
**Solution:** Use `sudo` (requires root privileges to capture packets)
```bash
sudo python src/real_time_detection.py
```

### Issue: "Dataset not found" error
**Solution:** Download the dataset first
```bash
python data/download_dataset.py
```

### Issue: Out of memory when training
**Solution:** Use a smaller dataset or increase available RAM
```bash
# Use only first 100k rows
python src/model_training.py --sample 100000
```

## 📚 References & Resources

### Datasets
- [UNSW-NB15](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/)
- [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)
- [KDD Cup 99](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)

### Papers & Documentation
- "UNSW-NB15: a comprehensive data set for network intrusion detection systems" - 2015
- MITRE ATT&CK Framework: https://attack.mitre.org/
- scikit-learn documentation: https://scikit-learn.org/
- XGBoost documentation: https://xgboost.readthedocs.io/
- TensorFlow/Keras: https://www.tensorflow.org/

### Cybersecurity Concepts
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [CIS Controls](https://www.cisecurity.org/controls/)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**blackcop1** - Cybersecurity & ML enthusiast

## 📞 Contact & Support

- **GitHub Issues:** Open an issue for bugs or features
- **Discussions:** Share ideas and ask questions
- **Email:** Contact via GitHub profile

## 🎯 Future Enhancements

- [ ] Add more attack types (APT, zero-day detection)
- [ ] Implement ensemble methods combining all models
- [ ] Deploy as REST API for easy integration
- [ ] Create web dashboard for visualization
- [ ] Add Kubernetes deployment manifests
- [ ] Implement automated model retraining
- [ ] Add explainability (SHAP values)
- [ ] Support for encrypted traffic analysis
- [ ] Integration with SIEM tools (Splunk, ELK)
- [ ] Mobile app for alerts

## ⭐ If You Find This Helpful

Please consider giving this repository a star! It helps other cybersecurity enthusiasts discover this project.

---

**Happy Learning! 🚀** This project is perfect for cybersecurity interviews and placements. Good luck!
