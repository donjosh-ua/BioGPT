"""
Configuration settings for the BioGPT project.
"""

import os
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ModelConfig:
    """Configuration for model parameters"""

    base_model_name: str = "DeepESP/gpt2-spanish"
    model_save_path: str = "./biogpt_model"
    max_length: int = 512
    learning_rate: float = 5e-5
    batch_size: int = 4
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100


@dataclass
class DataConfig:
    """Configuration for data handling"""

    data_dir: str = "./data"
    scraped_data_dir: str = "./data/scraped_data"
    processed_data_dir: str = "./data/processed"
    training_file: str = "training_data.txt"
    validation_file: str = "validation_data.txt"
    test_file: str = "test_data.txt"

    # Control tokens for different difficulty levels
    level_tokens: Dict[str, str] = None

    def __post_init__(self):
        if self.level_tokens is None:
            self.level_tokens = {
                "principiante": "[PRINCIPIANTE]",
                "intermedio": "[INTERMEDIO]",
                "experto": "[EXPERTO]",
            }

        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.scraped_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)


@dataclass
class ScrapingConfig:
    """Configuration for web scraping"""

    max_pages_per_source: int = 100
    delay_between_requests: float = 1.0
    timeout: int = 10
    headers: Dict[str, str] = None

    def __post_init__(self):
        if self.headers is None:
            self.headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation"""

    test_concepts: List[str] = None
    evaluation_levels: List[str] = None

    def __post_init__(self):
        if self.test_concepts is None:
            self.test_concepts = [
                "célula",
                "homeostasis",
                "CRISPR",
                "potencial de acción",
                "sinapsis",
                "fotosíntesis",
                "respiración celular",
                "sistema cardiovascular",
                "neurona",
                "ADN",
            ]

        if self.evaluation_levels is None:
            self.evaluation_levels = ["principiante", "intermedio", "experto"]


# Global configuration instance
model_config = ModelConfig()
data_config = DataConfig()
scraping_config = ScrapingConfig()
evaluation_config = EvaluationConfig()
