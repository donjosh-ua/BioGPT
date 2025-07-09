#!/usr/bin/env python3
"""
Demo script for BioGPT model.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.model_evaluator import ModelEvaluator
from src.config import evaluation_config
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_model():
    """Run a demo of the BioGPT model"""
    logger.info("Starting BioGPT Demo...")

    # Initialize evaluator
    evaluator = ModelEvaluator()

    print("=" * 60)
    print("🧬 BioLearn-GPT Demo")
    print("=" * 60)

    # Test concepts at different levels
    test_cases = [
        ("célula", "principiante"),
        ("homeostasis", "intermedio"),
        ("transducción de señales", "experto"),
        ("fotosíntesis", "principiante"),
        ("ADN", "intermedio"),
        ("regulación génica", "experto"),
    ]

    for concept, level in test_cases:
        print(f"\n{'='*50}")
        print(f"Concepto: {concept.title()}")
        print(f"Nivel: {level.title()}")
        print(f"{'='*50}")

        # Create prompt
        level_token = f"[{level.upper()}]"
        prompt = f"{level_token} Explica el concepto: {concept}"

        try:
            # Generate explanation
            explanation = evaluator.generate_text(prompt, max_length=200)
            print("\n📝 Explicación generada:")
            print(f"{explanation}")

            # Analyze complexity
            analysis = evaluator._analyze_text_complexity(explanation)
            print("\n📊 Análisis:")
            print(
                f"   • Longitud promedio de oración: {analysis['avg_sentence_length']:.1f} palabras"
            )
            print(f"   • Términos técnicos: {analysis['technical_terms_count']}")
            print(f"   • Nivel estimado: {analysis['estimated_level']}")
            print(f"   • Total de palabras: {analysis['word_count']}")

        except Exception as e:
            print(f"❌ Error generando explicación: {e}")

        print("-" * 50)

    print("\n🎉 Demo completado!")
    print("Para usar la interfaz web, ejecuta: python src/gradio_app.py")


if __name__ == "__main__":
    demo_model()
