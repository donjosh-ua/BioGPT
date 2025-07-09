"""
Model evaluation and analysis module.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
import logging
from typing import Dict, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import model_config, evaluation_config
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluator class for comprehensive model analysis"""

    def __init__(self, model_path: str = None):
        self.model_path = model_path or model_config.model_save_path
        self.model = None
        self.tokenizer = None
        self.evaluation_results = {}

    def load_model(self) -> None:
        """Load the trained model and tokenizer"""
        logger.info(f"Loading model from {self.model_path}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Falling back to base model...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_config.base_model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_config.base_model_name
            )

    def generate_text(
        self, prompt: str, max_length: int = 250, temperature: float = 0.7
    ) -> str:
        """
        Generate text from a given prompt

        Args:
            prompt: Input prompt
            max_length: Maximum length of generated text
            temperature: Temperature for generation

        Returns:
            Generated text
        """
        if self.model is None:
            self.load_model()

        inputs = self.tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                num_return_sequences=1,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove prompt from output
        if prompt in generated_text:
            generated_text = generated_text.replace(prompt, "").strip()

        return generated_text

    def evaluate_level_consistency(self) -> Dict:
        """
        Evaluate how well the model responds to different difficulty levels

        Returns:
            Dictionary with consistency evaluation results
        """
        logger.info("Evaluating level consistency...")

        results = {}

        for concept in evaluation_config.test_concepts:
            results[concept] = {}

            for level in evaluation_config.evaluation_levels:
                # Create prompt
                level_token = f"[{level.upper()}]"
                prompt = f"{level_token} Explica el concepto: {concept}"

                # Generate text
                generated_text = self.generate_text(prompt)

                # Analyze the text
                analysis = self._analyze_text_complexity(generated_text)
                analysis["generated_text"] = generated_text
                analysis["prompt"] = prompt

                results[concept][level] = analysis

        self.evaluation_results["level_consistency"] = results
        return results

    def _analyze_text_complexity(self, text: str) -> Dict:
        """
        Analyze text complexity metrics

        Args:
            text: Text to analyze

        Returns:
            Dictionary with complexity metrics
        """
        if not text:
            return {
                "avg_sentence_length": 0,
                "avg_word_length": 0,
                "technical_terms_count": 0,
                "sentence_count": 0,
                "word_count": 0,
                "estimated_level": "unknown",
            }

        # Basic metrics
        sentences = text.split(".")
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()

        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0

        # Technical terms (basic list)
        technical_terms = [
            "proteína",
            "enzima",
            "mitocondria",
            "nucleótido",
            "genoma",
            "fosforilación",
            "transcripción",
            "traducción",
            "cromosoma",
            "metabolismo",
            "homeostasis",
            "transducción",
            "citoplasma",
            "ribosoma",
            "membrana",
            "célula",
            "orgánulo",
            "tejido",
            "sistema",
            "función",
            "proceso",
            "estructura",
            "molecular",
        ]

        technical_count = sum(1 for term in technical_terms if term in text.lower())

        # Estimate level based on complexity
        if avg_sentence_length > 25 and technical_count > 8:
            estimated_level = "experto"
        elif avg_sentence_length > 15 and technical_count > 4:
            estimated_level = "intermedio"
        else:
            estimated_level = "principiante"

        return {
            "avg_sentence_length": avg_sentence_length,
            "avg_word_length": avg_word_length,
            "technical_terms_count": technical_count,
            "sentence_count": len(sentences),
            "word_count": len(words),
            "estimated_level": estimated_level,
        }

    def evaluate_content_quality(self) -> Dict:
        """
        Evaluate content quality through various metrics

        Returns:
            Dictionary with quality metrics
        """
        logger.info("Evaluating content quality...")

        quality_results = {}

        # Test prompts for quality evaluation
        test_prompts = [
            ("[PRINCIPIANTE] Explica qué es una célula", "célula"),
            ("[INTERMEDIO] Explica el proceso de homeostasis", "homeostasis"),
            (
                "[EXPERTO] Explica el mecanismo de transducción de señales",
                "transducción",
            ),
            ("[PRINCIPIANTE] Explica qué es el ADN", "ADN"),
            ("[INTERMEDIO] Explica la fotosíntesis", "fotosíntesis"),
            ("[EXPERTO] Explica la regulación génica", "regulación génica"),
        ]

        for prompt, topic in test_prompts:
            generated_text = self.generate_text(prompt)

            # Quality metrics
            quality_metrics = {
                "coherence_score": self._calculate_coherence_score(generated_text),
                "relevance_score": self._calculate_relevance_score(
                    generated_text, topic
                ),
                "completeness_score": self._calculate_completeness_score(
                    generated_text
                ),
                "accuracy_indicators": self._check_accuracy_indicators(
                    generated_text, topic
                ),
                "generated_text": generated_text,
            }

            quality_results[f"{topic}_{prompt.split('[')[1].split(']')[0].lower()}"] = (
                quality_metrics
            )

        self.evaluation_results["content_quality"] = quality_results
        return quality_results

    def _calculate_coherence_score(self, text: str) -> float:
        """
        Calculate a simple coherence score based on text structure

        Args:
            text: Text to analyze

        Returns:
            Coherence score (0-1)
        """
        if not text:
            return 0.0

        # Simple heuristics for coherence
        sentences = text.split(".")
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) < 2:
            return 0.5

        # Check for logical flow indicators
        flow_indicators = [
            "además",
            "por tanto",
            "sin embargo",
            "también",
            "asimismo",
            "es decir",
            "por ejemplo",
        ]
        flow_count = sum(
            1 for indicator in flow_indicators if indicator in text.lower()
        )

        # Check for repetition (negative indicator)
        words = text.lower().split()
        unique_words = set(words)
        repetition_ratio = len(unique_words) / len(words) if words else 0

        # Combine metrics
        coherence_score = min(1.0, (flow_count * 0.1) + (repetition_ratio * 0.7) + 0.3)

        return coherence_score

    def _calculate_relevance_score(self, text: str, topic: str) -> float:
        """
        Calculate relevance score based on topic keywords

        Args:
            text: Generated text
            topic: Expected topic

        Returns:
            Relevance score (0-1)
        """
        if not text:
            return 0.0

        # Topic-specific keywords
        topic_keywords = {
            "célula": ["célula", "membrana", "núcleo", "citoplasma", "orgánulo"],
            "homeostasis": ["homeostasis", "equilibrio", "regulación", "estabilidad"],
            "transducción": ["señal", "transducción", "receptor", "proteína"],
            "ADN": ["ADN", "ácido", "nucleótido", "gen", "cromosoma"],
            "fotosíntesis": ["fotosíntesis", "luz", "clorofila", "glucosa", "oxígeno"],
            "regulación génica": ["gen", "regulación", "transcripción", "expresión"],
        }

        keywords = topic_keywords.get(topic, [topic])

        # Count keyword occurrences
        text_lower = text.lower()
        keyword_count = sum(1 for keyword in keywords if keyword in text_lower)

        # Calculate relevance score
        relevance_score = min(1.0, keyword_count / len(keywords))

        return relevance_score

    def _calculate_completeness_score(self, text: str) -> float:
        """
        Calculate completeness score based on text length and structure

        Args:
            text: Text to analyze

        Returns:
            Completeness score (0-1)
        """
        if not text:
            return 0.0

        # Basic completeness metrics
        word_count = len(text.split())
        sentence_count = len([s for s in text.split(".") if s.strip()])

        # Score based on length and structure
        length_score = min(1.0, word_count / 100)  # Ideal ~100 words
        structure_score = min(1.0, sentence_count / 5)  # Ideal ~5 sentences

        completeness_score = (length_score + structure_score) / 2

        return completeness_score

    def _check_accuracy_indicators(self, text: str, topic: str) -> Dict:
        """
        Check for accuracy indicators and potential issues

        Args:
            text: Generated text
            topic: Topic being discussed

        Returns:
            Dictionary with accuracy indicators
        """
        indicators = {
            "has_specific_terms": False,
            "has_contradictions": False,
            "has_vague_statements": False,
            "factual_confidence": 0.5,  # Default neutral
        }

        if not text:
            return indicators

        text_lower = text.lower()

        # Check for specific scientific terms
        scientific_terms = ["proceso", "función", "estructura", "mecanismo", "sistema"]
        indicators["has_specific_terms"] = any(
            term in text_lower for term in scientific_terms
        )

        # Check for vague statements
        vague_terms = [
            "probablemente",
            "tal vez",
            "puede ser",
            "quizás",
            "posiblemente",
        ]
        indicators["has_vague_statements"] = any(
            term in text_lower for term in vague_terms
        )

        # Simple factual confidence based on presence of definitive statements
        definitive_terms = ["es", "son", "tiene", "contiene", "realiza", "produce"]
        definitive_count = sum(1 for term in definitive_terms if term in text_lower)
        indicators["factual_confidence"] = min(1.0, definitive_count / 5)

        return indicators

    def generate_evaluation_report(self) -> str:
        """
        Generate a comprehensive evaluation report

        Returns:
            Formatted evaluation report
        """
        logger.info("Generating evaluation report...")

        if not self.evaluation_results:
            # Run evaluations if not done yet
            self.evaluate_level_consistency()
            self.evaluate_content_quality()

        report = []
        report.append("=" * 50)
        report.append("BIOGPT MODEL EVALUATION REPORT")
        report.append("=" * 50)

        # Level consistency results
        if "level_consistency" in self.evaluation_results:
            report.append("\n1. LEVEL CONSISTENCY EVALUATION")
            report.append("-" * 30)

            for concept, levels in self.evaluation_results["level_consistency"].items():
                report.append(f"\nConcept: {concept}")
                for level, analysis in levels.items():
                    report.append(f"  {level.title()}:")
                    report.append(
                        f"    - Avg sentence length: {analysis['avg_sentence_length']:.1f}"
                    )
                    report.append(
                        f"    - Technical terms: {analysis['technical_terms_count']}"
                    )
                    report.append(
                        f"    - Estimated level: {analysis['estimated_level']}"
                    )

        # Content quality results
        if "content_quality" in self.evaluation_results:
            report.append("\n2. CONTENT QUALITY EVALUATION")
            report.append("-" * 30)

            for test_case, metrics in self.evaluation_results[
                "content_quality"
            ].items():
                report.append(f"\nTest case: {test_case}")
                report.append(f"  - Coherence score: {metrics['coherence_score']:.2f}")
                report.append(f"  - Relevance score: {metrics['relevance_score']:.2f}")
                report.append(
                    f"  - Completeness score: {metrics['completeness_score']:.2f}"
                )
                report.append(
                    f"  - Factual confidence: {metrics['accuracy_indicators']['factual_confidence']:.2f}"
                )

        report_text = "\n".join(report)

        # Save report
        report_path = os.path.join(self.model_path, "evaluation_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        logger.info(f"Evaluation report saved to {report_path}")

        return report_text

    def save_evaluation_results(self) -> None:
        """Save evaluation results to JSON file"""
        results_path = os.path.join(self.model_path, "detailed_evaluation_results.json")

        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(self.evaluation_results, f, indent=2, ensure_ascii=False)

        logger.info(f"Detailed evaluation results saved to {results_path}")


def main():
    """Main function for running evaluation"""
    evaluator = ModelEvaluator()

    # Run comprehensive evaluation
    evaluator.evaluate_level_consistency()
    evaluator.evaluate_content_quality()

    # Generate and save report
    report = evaluator.generate_evaluation_report()
    evaluator.save_evaluation_results()

    print(report)


if __name__ == "__main__":
    main()
