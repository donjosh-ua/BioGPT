"""
Model training module for fine-tuning the GPT-2 model on bioengineering content.
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from datasets import Dataset
import os
import json
import logging
from typing import Dict, List, Optional
from config import model_config, data_config
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BioGPTTrainer:
    """Trainer class for fine-tuning GPT-2 on bioengineering content"""

    def __init__(self):
        self.model_config = model_config
        self.data_config = data_config
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def load_tokenizer_and_model(self) -> None:
        """Load tokenizer and model"""
        logger.info(f"Loading tokenizer and model: {self.model_config.base_model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.base_model_name
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config.base_model_name
        )

        # Add special tokens for our control tokens
        special_tokens = list(self.data_config.level_tokens.values()) + ["[FIN_TEXTO]"]
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Resize model embeddings to accommodate new tokens
        self.model.resize_token_embeddings(len(self.tokenizer))

        logger.info(f"Model loaded successfully. Vocab size: {len(self.tokenizer)}")

    def load_training_data(self) -> None:
        """Load and tokenize training data"""
        logger.info("Loading training data...")

        # Load training data
        train_path = os.path.join(
            self.data_config.processed_data_dir, self.data_config.training_file
        )
        val_path = os.path.join(
            self.data_config.processed_data_dir, self.data_config.validation_file
        )
        test_path = os.path.join(
            self.data_config.processed_data_dir, self.data_config.test_file
        )

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training data not found at {train_path}")

        # Read training data
        with open(train_path, "r", encoding="utf-8") as f:
            train_texts = f.read().strip().split("\n")

        # Read validation data
        val_texts = []
        if os.path.exists(val_path):
            with open(val_path, "r", encoding="utf-8") as f:
                val_texts = f.read().strip().split("\n")

        # Read test data
        test_texts = []
        if os.path.exists(test_path):
            with open(test_path, "r", encoding="utf-8") as f:
                test_texts = f.read().strip().split("\n")

        logger.info(f"Loaded {len(train_texts)} training examples")
        logger.info(f"Loaded {len(val_texts)} validation examples")
        logger.info(f"Loaded {len(test_texts)} test examples")

        # Tokenize data
        self.train_dataset = self._tokenize_data(train_texts)
        if val_texts:
            self.val_dataset = self._tokenize_data(val_texts)
        if test_texts:
            self.test_dataset = self._tokenize_data(test_texts)

    def _tokenize_data(self, texts: List[str]) -> Dataset:
        """
        Tokenize text data for training

        Args:
            texts: List of text strings

        Returns:
            Tokenized dataset
        """

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=self.model_config.max_length,
                return_tensors="pt",
            )

        # Create dataset
        dataset = Dataset.from_dict({"text": texts})

        # Tokenize
        tokenized_dataset = dataset.map(
            tokenize_function, batched=True, remove_columns=dataset.column_names
        )

        return tokenized_dataset

    def setup_training_arguments(self) -> TrainingArguments:
        """Setup training arguments"""
        logger.info("Setting up training arguments...")

        training_args = TrainingArguments(
            output_dir=self.model_config.model_save_path,
            overwrite_output_dir=True,
            num_train_epochs=self.model_config.num_epochs,
            per_device_train_batch_size=self.model_config.batch_size,
            per_device_eval_batch_size=self.model_config.batch_size,
            gradient_accumulation_steps=self.model_config.gradient_accumulation_steps,
            warmup_steps=self.model_config.warmup_steps,
            weight_decay=self.model_config.weight_decay,
            logging_dir=os.path.join(self.model_config.model_save_path, "logs"),
            logging_steps=self.model_config.logging_steps,
            save_steps=self.model_config.save_steps,
            eval_steps=self.model_config.eval_steps,
            eval_strategy="steps" if self.val_dataset else "no",
            save_strategy="steps",
            load_best_model_at_end=True if self.val_dataset else False,
            metric_for_best_model="eval_loss" if self.val_dataset else None,
            greater_is_better=False,
            save_total_limit=3,
            prediction_loss_only=True,
            learning_rate=self.model_config.learning_rate,
            lr_scheduler_type="linear",
            report_to=None,  # Disable wandb/tensorboard logging
            dataloader_pin_memory=False,
            fp16=torch.cuda.is_available(),  # Use mixed precision if CUDA available
        )

        return training_args

    def train_model(self) -> None:
        """Train the model"""
        logger.info("Starting model training...")

        if self.model is None or self.tokenizer is None:
            self.load_tokenizer_and_model()

        if self.train_dataset is None:
            self.load_training_data()

        # Setup training arguments
        training_args = self.setup_training_arguments()

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal language modeling, not masked LM
        )

        # Setup trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator,
            callbacks=(
                [EarlyStoppingCallback(early_stopping_patience=3)]
                if self.val_dataset
                else None
            ),
        )

        # Train the model
        trainer.train()

        # Save the final model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.model_config.model_save_path)

        logger.info(
            f"Model training completed and saved to {self.model_config.model_save_path}"
        )

    def evaluate_model(self) -> Dict:
        """
        Evaluate the trained model

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating model...")

        if self.test_dataset is None:
            logger.warning("No test dataset available for evaluation")
            return {}

        # Load the trained model
        model = AutoModelForCausalLM.from_pretrained(self.model_config.model_save_path)
        tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_save_path)

        # Setup evaluation arguments
        eval_args = TrainingArguments(
            output_dir=self.model_config.model_save_path,
            per_device_eval_batch_size=self.model_config.batch_size,
            dataloader_pin_memory=False,
            report_to=None,
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

        # Setup trainer for evaluation
        trainer = Trainer(
            model=model,
            args=eval_args,
            eval_dataset=self.test_dataset,
            data_collator=data_collator,
        )

        # Evaluate
        eval_results = trainer.evaluate()

        # Calculate perplexity
        perplexity = math.exp(eval_results["eval_loss"])
        eval_results["perplexity"] = perplexity

        logger.info(f"Evaluation results: {eval_results}")

        # Save evaluation results
        results_path = os.path.join(
            self.model_config.model_save_path, "evaluation_results.json"
        )
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, indent=2)

        return eval_results

    def generate_sample_outputs(
        self, test_prompts: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Generate sample outputs for qualitative evaluation

        Args:
            test_prompts: List of test prompts. If None, uses default prompts.

        Returns:
            List of generated samples
        """
        logger.info("Generating sample outputs...")

        if test_prompts is None:
            test_prompts = [
                "[PRINCIPIANTE] Explica qué es una célula",
                "[INTERMEDIO] Explica el proceso de homeostasis",
                "[EXPERTO] Explica el mecanismo de transducción de señales",
                "[PRINCIPIANTE] Explica qué es el ADN",
                "[INTERMEDIO] Explica la fotosíntesis",
                "[EXPERTO] Explica la regulación génica",
            ]

        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(self.model_config.model_save_path)
        tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_save_path)

        samples = []

        for prompt in test_prompts:
            # Generate text
            inputs = tokenizer(prompt, return_tensors="pt")

            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    max_length=300,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove prompt from output
            if prompt in generated_text:
                generated_text = generated_text.replace(prompt, "").strip()

            samples.append({"prompt": prompt, "generated_text": generated_text})

        # Save samples
        samples_path = os.path.join(
            self.model_config.model_save_path, "sample_outputs.json"
        )
        with open(samples_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)

        logger.info(f"Generated {len(samples)} sample outputs")

        return samples


def main():
    """Main function for training"""
    trainer = BioGPTTrainer()

    # Train the model
    trainer.train_model()

    # Evaluate the model
    eval_results = trainer.evaluate_model()

    # Generate sample outputs
    samples = trainer.generate_sample_outputs()

    logger.info("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()
