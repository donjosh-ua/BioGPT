"""
Gradio web interface for BioGPT model.

This module provides a user-friendly web interface for interacting with the trained BioGPT model.
Users can input bioengineering concepts and select difficulty levels to generate educational content.
"""

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import logging
from typing import Optional
from config import model_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BioGPTGenerator:
    """Generator class for BioGPT model interface"""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or model_config.model_save_path
        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        """Load the fine-tuned model and tokenizer"""
        try:
            logger.info(f"Loading model from {self.model_path}")

            # Check if fine-tuned model exists
            if os.path.exists(self.model_path):
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
                logger.info("Fine-tuned model loaded successfully!")
            else:
                raise FileNotFoundError(f"Model not found at {self.model_path}")

        except Exception as e:
            logger.error(f"Error loading fine-tuned model: {e}")
            logger.info("Falling back to base model...")

            # Fallback to base model
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_config.base_model_name
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_config.base_model_name
                )
                logger.info("Base model loaded successfully!")
            except Exception as base_error:
                logger.error(f"Error loading base model: {base_error}")
                raise

    def generate_text(
        self, tema: str, nivel: str, max_length: int = 250, temperature: float = 0.7
    ) -> str:
        """
        Generate explanation based on topic and level

        Args:
            tema: Topic to explain
            nivel: Difficulty level (Principiante, Intermedio, Experto)
            max_length: Maximum length of generated text
            temperature: Temperature for generation (0.1-1.0)

        Returns:
            Generated explanation text
        """
        if not tema.strip():
            return "Por favor, introduce un concepto para explicar."

        # Map level to control token
        level_map = {
            "Principiante": "[PRINCIPIANTE]",
            "Intermedio": "[INTERMEDIO]",
            "Experto": "[EXPERTO]",
        }

        level_token = level_map.get(nivel, "[INTERMEDIO]")
        prompt = f"{level_token} Explica el siguiente concepto: {tema}\n\n"

        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt")

            # Generate text
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=3,
                )

            # Decode output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Clean up the output
            generated_text = self._clean_output(generated_text, prompt)

            return generated_text

        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return f"Error generando el texto: {str(e)}"

    def _clean_output(self, generated_text: str, prompt: str) -> str:
        """
        Clean and format the generated output

        Args:
            generated_text: Raw generated text
            prompt: Original prompt

        Returns:
            Cleaned output text
        """
        # Remove the prompt from the output
        if prompt in generated_text:
            generated_text = generated_text.replace(prompt, "").strip()

        # Remove any remaining control tokens
        control_tokens = ["[PRINCIPIANTE]", "[INTERMEDIO]", "[EXPERTO]", "[FIN_TEXTO]"]
        for token in control_tokens:
            generated_text = generated_text.replace(token, "")

        # Clean up formatting
        generated_text = generated_text.strip()

        # Ensure it ends properly
        if generated_text and not generated_text.endswith((".", "!", "?")):
            generated_text += "."

        return generated_text


# Initialize the generator
try:
    generator = BioGPTGenerator()
    model_loaded = True
except Exception as e:
    logger.error(f"Failed to initialize generator: {e}")
    generator = None
    model_loaded = False


def generar_explicacion(
    tema: str, nivel: str, longitud: int = 250, temperatura: float = 0.7
) -> str:
    """
    Wrapper function for Gradio interface

    Args:
        tema: Topic to explain
        nivel: Difficulty level
        longitud: Maximum length of output
        temperatura: Temperature for generation

    Returns:
        Generated explanation or error message
    """
    if not model_loaded or generator is None:
        return "❌ Error: El modelo no se pudo cargar. Por favor, entrena el modelo primero ejecutando 'python main.py'."

    if not tema.strip():
        return "⚠️ Por favor, introduce un concepto para explicar."

    try:
        return generator.generate_text(tema, nivel, longitud, temperatura)
    except Exception as e:
        return f"❌ Error generando la explicación: {str(e)}"


def get_model_info() -> str:
    """Get information about the loaded model"""
    if not model_loaded or generator is None:
        return "❌ Modelo no cargado"

    try:
        model_path = generator.model_path
        if os.path.exists(model_path):
            return f"✅ Modelo fine-tuneado cargado desde: {model_path}"
        else:
            return f"⚠️ Modelo base cargado: {model_config.base_model_name}"
    except:
        return "❓ Estado del modelo desconocido"


# Create Gradio interface
with gr.Blocks(title="BioLearn-GPT", theme=gr.themes.Soft()) as iface:
    gr.HTML(
        """
    <div style="text-align: center; padding: 20px;">
        <h1>🧬 BioLearn-GPT</h1>
        <h3>Generador de Contenido Educativo en Bioingeniería</h3>
        <p>Introduce un concepto y selecciona tu nivel de conocimiento para obtener una explicación personalizada.</p>
    </div>
    """
    )

    with gr.Row():
        with gr.Column(scale=2):
            # Input section
            gr.Markdown("### 📝 Configuración")

            tema_input = gr.Textbox(
                lines=2,
                placeholder="Ejemplos: célula, homeostasis, CRISPR, sinapsis, fotosíntesis...",
                label="Concepto de Bioingeniería/Fisiología",
                info="Escribe el concepto que quieres que se explique",
            )

            nivel_input = gr.Dropdown(
                choices=["Principiante", "Intermedio", "Experto"],
                value="Principiante",
                label="Nivel de Conocimiento",
                info="Selecciona tu nivel de conocimiento sobre el tema",
            )

            # Advanced options
            with gr.Accordion("⚙️ Opciones Avanzadas", open=False):
                longitud_input = gr.Slider(
                    minimum=100,
                    maximum=500,
                    value=250,
                    step=50,
                    label="Longitud Máxima",
                    info="Longitud máxima del texto generado",
                )

                temperatura_input = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.7,
                    step=0.1,
                    label="Temperatura",
                    info="Controla la creatividad (0.1 = conservador, 1.0 = creativo)",
                )

            generar_btn = gr.Button(
                "🚀 Generar Explicación", variant="primary", size="lg"
            )

        with gr.Column(scale=3):
            # Output section
            gr.Markdown("### 📖 Explicación Generada")

            output_text = gr.Textbox(
                lines=10,
                label="Resultado",
                placeholder="La explicación aparecerá aquí...",
                show_copy_button=True,
            )

            # Model info
            model_info = gr.Textbox(
                value=get_model_info(),
                label="Estado del Modelo",
                interactive=False,
                max_lines=2,
            )

    # Examples section
    gr.Markdown("### 💡 Ejemplos de Uso")

    examples = gr.Examples(
        examples=[
            ["célula", "Principiante", 200, 0.7],
            ["homeostasis", "Intermedio", 250, 0.7],
            ["transducción de señales", "Experto", 300, 0.8],
            ["fotosíntesis", "Principiante", 200, 0.6],
            ["sistema cardiovascular", "Intermedio", 250, 0.7],
            ["regulación génica", "Experto", 300, 0.8],
        ],
        inputs=[tema_input, nivel_input, longitud_input, temperatura_input],
        outputs=output_text,
        fn=generar_explicacion,
        cache_examples=False,
    )

    # Event handlers
    generar_btn.click(
        fn=generar_explicacion,
        inputs=[tema_input, nivel_input, longitud_input, temperatura_input],
        outputs=output_text,
    )

    # Footer
    gr.HTML(
        """
    <div style="text-align: center; padding: 20px; margin-top: 30px; border-top: 1px solid #ddd;">
        <p><strong>BioLearn-GPT</strong> - Proyecto de Generación de Contenido Educativo</p>
        <p>Desarrollado con ❤️ usando Transformers y Gradio</p>
    </div>
    """
    )

if __name__ == "__main__":
    logger.info("Launching BioLearn-GPT interface...")

    # Launch the interface
    iface.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        quiet=False,
    )
