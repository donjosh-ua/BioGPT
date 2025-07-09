#!/usr/bin/env python3
"""
Test script to verify BioGPT installation and basic functionality.
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


class TestBioGPTInstallation(unittest.TestCase):
    """Test suite for BioGPT installation verification"""

    def test_import_dependencies(self):
        """Test that all required dependencies can be imported"""
        try:
            import torch
            import transformers
            import gradio
            import pandas
            import numpy
            import requests
            from bs4 import BeautifulSoup  # Import as bs4
            import nltk
            import sklearn

            self.assertTrue(True, "All dependencies imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import required dependency: {e}")

    def test_config_module(self):
        """Test that configuration module works"""
        try:
            from src.config import model_config, data_config

            self.assertIsNotNone(model_config.base_model_name)
            self.assertIsNotNone(data_config.data_dir)
            self.assertTrue(True, "Configuration module works")
        except Exception as e:
            self.fail(f"Configuration module failed: {e}")

    def test_data_processor(self):
        """Test data processor instantiation"""
        try:
            from src.data_processor import DataProcessor

            processor = DataProcessor()
            self.assertIsNotNone(processor)
            self.assertTrue(True, "Data processor instantiated successfully")
        except Exception as e:
            self.fail(f"Data processor failed: {e}")

    def test_web_scraper(self):
        """Test web scraper instantiation"""
        try:
            from src.web_scraper import WebScraper

            scraper = WebScraper()
            self.assertIsNotNone(scraper)
            self.assertTrue(True, "Web scraper instantiated successfully")
        except Exception as e:
            self.fail(f"Web scraper failed: {e}")

    @patch("src.model_trainer.AutoTokenizer")
    @patch("src.model_trainer.AutoModelForCausalLM")
    def test_model_trainer(self, mock_model, mock_tokenizer):
        """Test model trainer instantiation"""
        try:
            # Mock the tokenizer and model
            mock_tokenizer.from_pretrained.return_value = MagicMock()
            mock_model.from_pretrained.return_value = MagicMock()

            from src.model_trainer import BioGPTTrainer

            trainer = BioGPTTrainer()
            self.assertIsNotNone(trainer)
            self.assertTrue(True, "Model trainer instantiated successfully")
        except Exception as e:
            self.fail(f"Model trainer failed: {e}")

    @patch("src.model_evaluator.AutoTokenizer")
    @patch("src.model_evaluator.AutoModelForCausalLM")
    def test_model_evaluator(self, mock_model, mock_tokenizer):
        """Test model evaluator instantiation"""
        try:
            # Mock the tokenizer and model
            mock_tokenizer.from_pretrained.return_value = MagicMock()
            mock_model.from_pretrained.return_value = MagicMock()

            from src.model_evaluator import ModelEvaluator

            evaluator = ModelEvaluator()
            self.assertIsNotNone(evaluator)
            self.assertTrue(True, "Model evaluator instantiated successfully")
        except Exception as e:
            self.fail(f"Model evaluator failed: {e}")

    def test_directories_exist(self):
        """Test that required directories exist"""
        required_dirs = ["src", "data", "data/scraped_data", "biogpt_model"]

        for directory in required_dirs:
            self.assertTrue(
                os.path.exists(directory), f"Directory {directory} does not exist"
            )

    def test_required_files_exist(self):
        """Test that required files exist"""
        required_files = [
            "requirements.txt",
            "README.md",
            "src/main.py",
            "src/config.py",
            "src/gradio_app.py",
        ]

        for file_path in required_files:
            self.assertTrue(
                os.path.exists(file_path), f"File {file_path} does not exist"
            )


def run_system_check():
    """Run a comprehensive system check"""
    print("🔍 Running BioGPT System Check...")
    print("=" * 50)

    # Check Python version
    print(f"Python version: {sys.version}")

    # Check if GPU is available
    try:
        import torch

        if torch.cuda.is_available():
            print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  GPU not available - will use CPU")
    except ImportError:
        print("❌ PyTorch not installed")

    # Check transformers version
    try:
        import transformers

        print(f"✅ Transformers version: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers not installed")

    # Check gradio version
    try:
        import gradio

        print(f"✅ Gradio version: {gradio.__version__}")
    except ImportError:
        print("❌ Gradio not installed")

    print("\n" + "=" * 50)
    print("Running unit tests...")
    print("=" * 50)

    # Run unit tests
    unittest.main(verbosity=2, exit=False)

    print("\n🎉 System check completed!")
    print("If all tests passed, you're ready to run the BioGPT pipeline!")


if __name__ == "__main__":
    run_system_check()
