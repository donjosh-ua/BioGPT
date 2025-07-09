"""
Data processing module for cleaning and preparing training data.
"""

import pandas as pd
import os
import re
import logging
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from config import data_config

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Data processor for preparing training data"""

    def __init__(self):
        self.data_config = data_config
        self.level_tokens = data_config.level_tokens

    def load_scraped_data(self, filename: str) -> pd.DataFrame:
        """
        Load scraped data from CSV file

        Args:
            filename: Name of the CSV file to load

        Returns:
            DataFrame with scraped data
        """
        file_path = os.path.join(self.data_config.scraped_data_dir, filename)

        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return pd.DataFrame()

        try:
            df = pd.read_csv(file_path, encoding="utf-8")
            logger.info(f"Loaded {len(df)} records from {filename}")
            return df
        except Exception as e:
            logger.error(f"Error loading {filename}: {str(e)}")
            return pd.DataFrame()

    def clean_text_content(self, text: str) -> str:
        """
        Clean and normalize text content

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove URLs
        text = re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "",
            text,
        )

        # Remove email addresses
        text = re.sub(r"\S+@\S+", "", text)

        # Remove excessive punctuation
        text = re.sub(r"[\.]{2,}", ".", text)
        text = re.sub(r"[!]{2,}", "!", text)
        text = re.sub(r"[\?]{2,}", "?", text)

        # Remove parenthetical references like (2020), (Smith et al.)
        text = re.sub(r"\([^)]*\d{4}[^)]*\)", "", text)
        text = re.sub(r"\([^)]*et al[^)]*\)", "", text)

        # Remove citations like [1], [2-5]
        text = re.sub(r"\[[0-9\-,\s]*\]", "", text)

        # Clean up spacing
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        return text

    def segment_text(self, text: str, max_length: int = 400) -> List[str]:
        """
        Segment text into smaller chunks suitable for training

        Args:
            text: Text to segment
            max_length: Maximum length of each segment

        Returns:
            List of text segments
        """
        if not text:
            return []

        # Split into sentences
        sentences = sent_tokenize(text)

        segments = []
        current_segment = ""

        for sentence in sentences:
            # Check if adding this sentence would exceed max length
            if len(current_segment) + len(sentence) + 1 <= max_length:
                current_segment += " " + sentence if current_segment else sentence
            else:
                # Save current segment if it's substantial
                if len(current_segment) > 50:
                    segments.append(current_segment.strip())

                # Start new segment
                current_segment = sentence

        # Add the last segment
        if len(current_segment) > 50:
            segments.append(current_segment.strip())

        return segments

    def format_training_data(self, df: pd.DataFrame) -> List[str]:
        """
        Format data for training with control tokens

        Args:
            df: DataFrame with scraped data

        Returns:
            List of formatted training examples
        """
        training_examples = []

        for _, row in df.iterrows():
            content = self.clean_text_content(row["content"])

            if len(content) < 100:  # Skip very short content
                continue

            # Get difficulty level
            difficulty = row.get("difficulty_level", "intermedio").lower()

            # Map difficulty to token
            level_token = self.level_tokens.get(
                difficulty, self.level_tokens["intermedio"]
            )

            # Segment the content
            segments = self.segment_text(content)

            for segment in segments:
                # Format with control token
                formatted_example = f"{level_token} {segment} [FIN_TEXTO]"
                training_examples.append(formatted_example)

        logger.info(f"Generated {len(training_examples)} training examples")
        return training_examples

    def create_difficulty_balanced_dataset(
        self, training_examples: List[str]
    ) -> List[str]:
        """
        Create a balanced dataset across difficulty levels

        Args:
            training_examples: List of training examples

        Returns:
            Balanced list of training examples
        """
        # Separate examples by difficulty level
        level_examples = {"principiante": [], "intermedio": [], "experto": []}

        for example in training_examples:
            for level, token in self.level_tokens.items():
                if example.startswith(token):
                    level_examples[level].append(example)
                    break

        # Find the minimum count to balance
        min_count = min(
            len(examples) for examples in level_examples.values() if examples
        )

        if min_count == 0:
            logger.warning("No examples found for balancing")
            return training_examples

        # Balance the dataset
        balanced_examples = []
        for level, examples in level_examples.items():
            if examples:
                balanced_examples.extend(examples[:min_count])

        logger.info(f"Created balanced dataset with {len(balanced_examples)} examples")
        logger.info(f"Examples per level: {min_count}")

        return balanced_examples

    def split_data(
        self, examples: List[str], test_size: float = 0.2, val_size: float = 0.1
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Split data into train, validation, and test sets

        Args:
            examples: List of training examples
            test_size: Fraction of data for testing
            val_size: Fraction of data for validation

        Returns:
            Tuple of (train_examples, val_examples, test_examples)
        """
        if len(examples) < 10:
            logger.warning("Not enough examples for proper splitting")
            return examples, [], []

        # First split: separate test set
        train_val, test = train_test_split(
            examples, test_size=test_size, random_state=42
        )

        # Second split: separate validation from training
        val_ratio = val_size / (1 - test_size)
        train, val = train_test_split(train_val, test_size=val_ratio, random_state=42)

        logger.info(
            f"Data split - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}"
        )

        return train, val, test

    def save_processed_data(
        self, train_data: List[str], val_data: List[str], test_data: List[str]
    ) -> None:
        """
        Save processed data to files

        Args:
            train_data: Training examples
            val_data: Validation examples
            test_data: Test examples
        """
        # Ensure processed data directory exists
        os.makedirs(self.data_config.processed_data_dir, exist_ok=True)

        # Save training data
        train_path = os.path.join(
            self.data_config.processed_data_dir, self.data_config.training_file
        )
        with open(train_path, "w", encoding="utf-8") as f:
            f.write("\n".join(train_data))

        # Save validation data
        val_path = os.path.join(
            self.data_config.processed_data_dir, self.data_config.validation_file
        )
        with open(val_path, "w", encoding="utf-8") as f:
            f.write("\n".join(val_data))

        # Save test data
        test_path = os.path.join(
            self.data_config.processed_data_dir, self.data_config.test_file
        )
        with open(test_path, "w", encoding="utf-8") as f:
            f.write("\n".join(test_data))

        logger.info(f"Saved processed data to {self.data_config.processed_data_dir}")

    def process_all_scraped_data(self) -> None:
        """
        Process all scraped data files and create training dataset
        """
        logger.info("Starting data processing pipeline...")

        all_examples = []

        # Process all CSV files in scraped data directory
        for filename in os.listdir(self.data_config.scraped_data_dir):
            if filename.endswith(".csv"):
                df = self.load_scraped_data(filename)
                if not df.empty:
                    examples = self.format_training_data(df)
                    all_examples.extend(examples)

        if not all_examples:
            logger.error("No training examples generated!")
            return

        # Balance the dataset
        balanced_examples = self.create_difficulty_balanced_dataset(all_examples)

        # Split the data
        train_data, val_data, test_data = self.split_data(balanced_examples)

        # Save processed data
        self.save_processed_data(train_data, val_data, test_data)

        logger.info("Data processing pipeline completed!")

        # Print statistics
        logger.info(f"Total examples processed: {len(all_examples)}")
        logger.info(f"Balanced examples: {len(balanced_examples)}")
        logger.info(
            f"Final split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}"
        )


def main():
    """Main function for running data processing"""
    processor = DataProcessor()
    processor.process_all_scraped_data()


if __name__ == "__main__":
    main()
