"""Dataset download script for IDS project"""

import os
import requests
from pathlib import Path
import zipfile

def download_file(url, destination):
    """Download file from URL"""
    try:
        print(f'Downloading from {url}...')
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        percentage = (downloaded / total_size) * 100
                        print(f'Progress: {percentage:.1f}%', end='\r')
        
        print(f'\nDownload completed: {destination}')
        return True
    except Exception as e:
        print(f'Error downloading file: {str(e)}')
        return False

def download_unsw_nb15():
    """Download UNSW-NB15 dataset"""
    print('\n' + '='*60)
    print('Downloading UNSW-NB15 Dataset')
    print('='*60)
    
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # These URLs should be updated with actual download links
    urls = {
        'training': 'https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/UNSW-NB15_training-set.csv',
        'testing': 'https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/UNSW-NB15_testing-set.csv'
    }
    
    for dataset_type, url in urls.items():
        destination = data_dir / f'UNSW-NB15_{dataset_type}-set.csv'
        
        if destination.exists():
            print(f'{destination} already exists. Skipping...')
            continue
        
        print(f'\nDownloading {dataset_type} set...')
        if download_file(url, destination):
            print(f'✓ {dataset_type} set downloaded successfully')
        else:
            print(f'✗ Failed to download {dataset_type} set')
    
    print('\nDataset download completed!')

if __name__ == '__main__':
    try:
        download_unsw_nb15()
    except KeyboardInterrupt:
        print('\nDownload interrupted by user')
    except Exception as e:
        print(f'Error: {str(e)}')
