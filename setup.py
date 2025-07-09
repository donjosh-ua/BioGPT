#!/usr/bin/env python3
"""
Setup utility for BioGPT project.
"""

import os
import subprocess
import sys
import logging
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        return False
    logger.info(f"Python version: {sys.version}")
    return True


def install_requirements():
    """Install required packages"""
    try:
        logger.info("Installing requirements...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        )
        logger.info("Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing requirements: {e}")
        return False


def download_nltk_data():
    """Download required NLTK data"""
    try:
        logger.info("Downloading NLTK data...")
        import nltk

        nltk.download("punkt", quiet=True)
        logger.info("NLTK data downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error downloading NLTK data: {e}")
        return False


def create_directories():
    """Create necessary directories"""
    directories = ["data/scraped_data", "data/processed", "biogpt_model", "logs"]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def check_gpu_availability():
    """Check if GPU is available"""
    try:

        logger.info(f"PyTorch version: {torch.__version__}")

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info("CUDA available: Yes")
            logger.info(f"Number of GPUs: {gpu_count}")
            logger.info(f"CUDA version: {torch.version.cuda}")

            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = (
                    torch.cuda.get_device_properties(i).total_memory / 1024**3
                )  # GB
                logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            logger.warning("CUDA not available - using CPU")
            logger.info("Possible reasons:")
            logger.info("1. No NVIDIA GPU detected")
            logger.info("2. NVIDIA drivers not installed")
            logger.info("3. CUDA toolkit not installed")
            logger.info("4. PyTorch installed without CUDA support")

        # Additional checks
        if hasattr(torch.backends, "cudnn"):
            logger.info(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
            if torch.backends.cudnn.enabled:
                logger.info(f"cuDNN version: {torch.backends.cudnn.version()}")

    except ImportError:
        logger.warning("PyTorch not installed - cannot check GPU availability")
    except Exception as e:
        logger.error(f"Error checking GPU availability: {e}")

        # Try alternative GPU detection methods
        try:
            # Check nvidia-smi command
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("nvidia-smi command works - GPU hardware detected")
                logger.info("Issue might be with PyTorch CUDA installation")
            else:
                logger.info("nvidia-smi command failed - no NVIDIA GPU detected")
        except FileNotFoundError:
            logger.info("nvidia-smi not found - NVIDIA drivers might not be installed")
        except Exception as e2:
            logger.error(f"Error running nvidia-smi: {e2}")


def main():
    """Main setup function"""
    logger.info("Setting up BioGPT project...")

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Create directories
    create_directories()

    # Install requirements
    # if not install_requirements():
    #     sys.exit(1)

    # Download NLTK data
    if not download_nltk_data():
        logger.warning("NLTK data download failed - may need to download manually")

    # Check GPU availability
    check_gpu_availability()

    logger.info("Setup completed successfully!")
    logger.info("Next steps:")
    logger.info("1. Run 'python src/main.py --mode all' to start the complete pipeline")
    logger.info(
        "2. Or run 'python src/main.py --mode scrape' to start with data collection"
    )
    logger.info("3. Launch the web interface with 'python src/gradio_app.py'")


if __name__ == "__main__":
    main()
