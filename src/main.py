"""
Main pipeline for BioGPT project.

This module orchestrates the complete pipeline:
1. Data scraping and collection
2. Data processing and preparation
3. Model training and fine-tuning
4. Model evaluation and analysis
5. Gradio interface deployment

Usage:
    python main.py --mode [scrape|process|train|evaluate|all]
"""

import argparse
import logging
import os
import sys
from typing import Optional

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from web_scraper import WebScraper
from data_processor import DataProcessor
from model_trainer import BioGPTTrainer
from model_evaluator import ModelEvaluator
from config import model_config, data_config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BioGPTPipeline:
    """Main pipeline class for BioGPT project"""

    def __init__(self):
        self.scraper = WebScraper()
        self.processor = DataProcessor()
        self.trainer = BioGPTTrainer()
        self.evaluator = ModelEvaluator()

    def run_data_scraping(self) -> None:
        """Run data scraping pipeline"""
        logger.info("Starting data scraping phase...")

        try:
            self.scraper.run_scraping_pipeline()
            logger.info("Data scraping completed successfully!")
        except Exception as e:
            logger.error(f"Error during data scraping: {e}")
            raise

    def run_data_processing(self) -> None:
        """Run data processing pipeline"""
        logger.info("Starting data processing phase...")

        try:
            self.processor.process_all_scraped_data()
            logger.info("Data processing completed successfully!")
        except Exception as e:
            logger.error(f"Error during data processing: {e}")
            raise

    def run_model_training(self) -> None:
        """Run model training pipeline"""
        logger.info("Starting model training phase...")

        try:
            # Check if processed data exists
            train_file = os.path.join(
                data_config.processed_data_dir, data_config.training_file
            )
            if not os.path.exists(train_file):
                logger.warning(
                    "Training data not found. Running data processing first..."
                )
                self.run_data_processing()

            # Train the model
            self.trainer.train_model()
            logger.info("Model training completed successfully!")

        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise

    def run_model_evaluation(self) -> None:
        """Run model evaluation pipeline"""
        logger.info("Starting model evaluation phase...")

        try:
            # Check if trained model exists
            if not os.path.exists(model_config.model_save_path):
                logger.warning("Trained model not found. Running training first...")
                self.run_model_training()

            # Evaluate the model
            self.evaluator.evaluate_level_consistency()
            self.evaluator.evaluate_content_quality()

            # Generate comprehensive report
            report = self.evaluator.generate_evaluation_report()
            self.evaluator.save_evaluation_results()

            logger.info("Model evaluation completed successfully!")
            print("\n" + "=" * 50)
            print("EVALUATION SUMMARY")
            print("=" * 50)
            print(report)

        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            raise

    def run_full_pipeline(self) -> None:
        """Run the complete pipeline"""
        logger.info("Starting complete BioGPT pipeline...")

        try:
            # Phase 1: Data Scraping
            logger.info("Phase 1/4: Data Scraping")
            self.run_data_scraping()

            # Phase 2: Data Processing
            logger.info("Phase 2/4: Data Processing")
            self.run_data_processing()

            # Phase 3: Model Training
            logger.info("Phase 3/4: Model Training")
            self.run_model_training()

            # Phase 4: Model Evaluation
            logger.info("Phase 4/4: Model Evaluation")
            self.run_model_evaluation()

            logger.info("Complete pipeline finished successfully!")

            # Show final summary
            self.show_pipeline_summary()

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

    def show_pipeline_summary(self) -> None:
        """Show pipeline completion summary"""
        print("\n" + "=" * 60)
        print("BIOGPT PIPELINE COMPLETION SUMMARY")
        print("=" * 60)

        # Check what was created
        summary_items = []

        # Scraped data
        scraped_files = []
        if os.path.exists(data_config.scraped_data_dir):
            scraped_files = [
                f
                for f in os.listdir(data_config.scraped_data_dir)
                if f.endswith(".csv")
            ]

        if scraped_files:
            summary_items.append(f"✓ Scraped data: {len(scraped_files)} files")

        # Processed data
        processed_files = []
        if os.path.exists(data_config.processed_data_dir):
            processed_files = [
                f
                for f in os.listdir(data_config.processed_data_dir)
                if f.endswith(".txt")
            ]

        if processed_files:
            summary_items.append(f"✓ Processed data: {len(processed_files)} files")

        # Trained model
        if os.path.exists(model_config.model_save_path):
            summary_items.append(f"✓ Trained model: {model_config.model_save_path}")

        # Evaluation results
        eval_files = []
        if os.path.exists(model_config.model_save_path):
            eval_files = [
                f
                for f in os.listdir(model_config.model_save_path)
                if f.startswith("evaluation")
            ]

        if eval_files:
            summary_items.append(f"✓ Evaluation results: {len(eval_files)} files")

        for item in summary_items:
            print(item)

        print("\nNext steps:")
        print("1. Run the Gradio interface: python gradio_app.py")
        print("2. Review evaluation results in the model directory")
        print("3. Test the model with different prompts")
        print("=" * 60)


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="BioGPT Pipeline")
    parser.add_argument(
        "--mode",
        choices=["scrape", "process", "train", "evaluate", "all"],
        default="all",
        help="Pipeline mode to run",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create pipeline
    pipeline = BioGPTPipeline()

    try:
        # Run selected mode
        if args.mode == "scrape":
            pipeline.run_data_scraping()
        elif args.mode == "process":
            pipeline.run_data_processing()
        elif args.mode == "train":
            pipeline.run_model_training()
        elif args.mode == "evaluate":
            pipeline.run_model_evaluation()
        elif args.mode == "all":
            pipeline.run_full_pipeline()

        print("\n🎉 Pipeline completed successfully!")

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
