# Data Directory

This directory contains datasets for the IDS project.

## Datasets

### UNSW-NB15
- **Source**: https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/
- **Size**: ~2.5 GB
- **Samples**: ~2.5 million records
- **Attack Types**: 10 categories
- **Features**: 49 features

### CICIDS2017
- **Source**: https://www.unb.ca/cic/datasets/ids-2017.html
- **Size**: ~80 GB
- **Samples**: ~2.8 million records
- **Attack Types**: 14 categories
- **Features**: 78 features

## Download Instructions

### For UNSW-NB15:
```bash
# Download from official source
wget https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/UNSW-NB15-training-set.csv
wget https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/UNSW-NB15-testing-set.csv
```

### Using download script:
```bash
python data/download_dataset.py
```

## Feature Description

### UNSW-NB15 Features:
- **srcip**: Source IP address
- **sport**: Source port number
- **dstip**: Destination IP address
- **dsport**: Destination port number
- **proto**: Protocol (TCP/UDP/ICMP)
- **state**: Connection state
- **dur**: Duration of connection
- **sbytes**: Source bytes
- **dbytes**: Destination bytes
- **sttl**: Source TTL
- **dttl**: Destination TTL
- **sloss**: Source packet loss
- **dloss**: Destination packet loss
- **service**: Service name
- **sload**: Source load
- **dload**: Destination load
- **spkts**: Source packets
- **dpkts**: Destination packets
- And more...

## Data Preparation

The `data_preprocessing.py` script will:
1. Load the CSV file
2. Handle missing values
3. Remove duplicates
4. Encode categorical features
5. Normalize numerical features
6. Split into train/test sets

## Notes

- For testing/development, consider using a subset of the data
- Ensure you have sufficient disk space (at least 5 GB free)
- Downloads may take time depending on internet speed
- Always verify file integrity after download
