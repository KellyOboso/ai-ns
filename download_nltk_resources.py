import nltk
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the NLTK data directory relative to the script
NLTK_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'nltk_data')

# Ensure the directory exists
os.makedirs(NLTK_DATA_DIR, exist_ok=True)

# Point NLTK to this directory
nltk.data.path.append(NLTK_DATA_DIR)
logger.info(f"NLTK data path set to: {NLTK_DATA_DIR}")

# List of NLTK packages to download
nltk_packages = ['punkt', 'stopwords', 'averaged_perceptron_tagger']

def download_nltk_package(package_name):
    try:
        nltk.data.find(f'tokenizers/{package_name}' if package_name == 'punkt' else f'corpora/{package_name}')
        logger.info(f"NLTK package '{package_name}' already downloaded.")
    except nltk.downloader.DownloadError:
        logger.info(f"Downloading NLTK package '{package_name}'...")
        try:
            nltk.download(package_name, download_dir=NLTK_DATA_DIR, quiet=False) # quiet=False for deployment visibility
            logger.info(f"Successfully downloaded NLTK package '{package_name}'.")
        except Exception as e:
            logger.error(f"Failed to download NLTK package '{package_name}': {e}")
            sys.exit(1) # Exit if critical download fails

if __name__ == "__main__":
    logger.info("Starting NLTK data download for Kelly AI.")
    for package in nltk_packages:
        download_nltk_package(package)
    logger.info("NLTK data download complete.")
