#!/usr/bin/env python3
"""
Usage guide and quickstart for BioGPT project.
"""

import os
import sys


def print_header(text, char="="):
    """Print a formatted header"""
    print(f"\n{char * 60}")
    print(f"{text:^60}")
    print(f"{char * 60}")


def print_section(text, char="-"):
    """Print a formatted section header"""
    print(f"\n{char * 40}")
    print(f"{text}")
    print(f"{char * 40}")


def show_quickstart():
    """Show quickstart guide"""
    print_header("🚀 BioGPT Quickstart Guide")

    print(
        """
¡Bienvenido a BioLearn-GPT!
Este sistema genera contenido educativo adaptativo para bioingeniería.
"""
    )

    print_section("1. Instalación Inicial")
    print(
        """
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\\Scripts\\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar setup
python setup.py
"""
    )

    print_section("2. Ejecución Rápida")
    print(
        """
# Opción 1: Pipeline completo (recomendado para primera ejecución)
python src/main.py --mode all

# Opción 2: Solo interfaz web (si ya tienes modelo entrenado)
python src/gradio_app.py

# Opción 3: Ejecutar demo
python demo.py
"""
    )

    print_section("3. Ejecución por Fases")
    print(
        """
# Fase 1: Recolección de datos
python src/main.py --mode scrape

# Fase 2: Procesamiento de datos
python src/main.py --mode process

# Fase 3: Entrenamiento del modelo
python src/main.py --mode train

# Fase 4: Evaluación del modelo
python src/main.py --mode evaluate
"""
    )

    print_section("4. Uso de la Interfaz Web")
    print(
        """
1. Ejecutar: python src/gradio_app.py
2. Abrir navegador en: http://localhost:7860
3. Introducir concepto (ej: "célula", "homeostasis")
4. Seleccionar nivel (Principiante, Intermedio, Experto)
5. Hacer clic en "Generar Explicación"
"""
    )


def show_detailed_usage():
    """Show detailed usage information"""
    print_header("📖 Uso Detallado del Sistema")

    print_section("Configuración Avanzada")
    print(
        """
Editar src/config.py para personalizar:

• ModelConfig: Parámetros del modelo
  - base_model_name: Modelo base a usar
  - learning_rate: Tasa de aprendizaje
  - batch_size: Tamaño de lote
  - num_epochs: Número de épocas

• DataConfig: Configuración de datos
  - data_dir: Directorio de datos
  - level_tokens: Tokens de control

• ScrapingConfig: Configuración de scraping
  - max_pages_per_source: Páginas máximas por fuente
  - delay_between_requests: Retraso entre peticiones
"""
    )

    print_section("Estructura de Datos")
    print(
        """
data/
├── scraped_data/          # Datos recolectados
│   ├── wikipedia_biology.csv
│   └── educational_sites.csv
├── processed/             # Datos procesados
│   ├── training_data.txt
│   ├── validation_data.txt
│   └── test_data.txt
└── ...

biogpt_model/              # Modelo entrenado
├── config.json
├── model.safetensors
├── tokenizer.json
├── evaluation_results.json
└── sample_outputs.json
"""
    )

    print_section("Formato de Prompts")
    print(
        """
El sistema usa tokens de control para niveles:

[PRINCIPIANTE] Texto simple y accesible
[INTERMEDIO] Texto con términos técnicos moderados
[EXPERTO] Texto técnico avanzado

Ejemplo de uso:
prompt = "[PRINCIPIANTE] Explica qué es una célula"
"""
    )

    print_section("Evaluación del Modelo")
    print(
        """
El sistema evalúa:

• Métricas Cuantitativas:
  - Perplejidad
  - Coherencia textual
  - Relevancia temática

• Métricas Cualitativas:
  - Consistencia de nivel
  - Precisión factual
  - Completitud educativa

Archivos de evaluación:
• evaluation_results.json: Métricas numéricas
• evaluation_report.txt: Reporte legible
• sample_outputs.json: Ejemplos generados
"""
    )


def show_troubleshooting():
    """Show troubleshooting guide"""
    print_header("🔧 Solución de Problemas")

    problems = [
        {
            "problem": "Error al cargar el modelo",
            "solution": """
• Verificar que el modelo esté entrenado:
  ls biogpt_model/
  
• Si no existe, ejecutar entrenamiento:
  python src/main.py --mode train
  
• Si persiste, usar modelo base:
  Se cargará automáticamente el modelo base
""",
        },
        {
            "problem": "Datos de entrenamiento no encontrados",
            "solution": """
• Ejecutar recolección de datos:
  python src/main.py --mode scrape
  
• Procesar datos:
  python src/main.py --mode process
  
• Verificar archivos:
  ls data/scraped_data/
  ls data/processed/
""",
        },
        {
            "problem": "Error de memoria durante entrenamiento",
            "solution": """
• Reducir batch_size en config.py
• Aumentar gradient_accumulation_steps
• Usar modelo más pequeño
• Verificar RAM/VRAM disponible
""",
        },
        {
            "problem": "Interfaz Gradio no carga",
            "solution": """
• Verificar puerto disponible (7860)
• Comprobar firewall
• Usar IP específica:
  iface.launch(server_name="0.0.0.0")
""",
        },
    ]

    for i, item in enumerate(problems, 1):
        print(f"\n{i}. {item['problem']}")
        print(f"   Solución:{item['solution']}")


def show_examples():
    """Show usage examples"""
    print_header("💡 Ejemplos de Uso")

    examples = [
        {
            "level": "Principiante",
            "concept": "célula",
            "expected": "Explicación simple usando lenguaje cotidiano",
        },
        {
            "level": "Intermedio",
            "concept": "homeostasis",
            "expected": "Explicación con términos técnicos moderados",
        },
        {
            "level": "Experto",
            "concept": "transducción de señales",
            "expected": "Explicación técnica detallada",
        },
    ]

    for example in examples:
        print(f"\n• Nivel: {example['level']}")
        print(f"  Concepto: {example['concept']}")
        print(f"  Resultado esperado: {example['expected']}")

    print_section("Conceptos Recomendados para Probar")
    concepts = [
        "célula",
        "homeostasis",
        "fotosíntesis",
        "respiración celular",
        "ADN",
        "ARN",
        "proteína",
        "enzima",
        "mitocondria",
        "núcleo",
        "membrana",
        "sistema nervioso",
        "neurona",
        "sinapsis",
        "potencial de acción",
        "sistema cardiovascular",
        "corazón",
        "metabolismo",
        "genética",
        "cromosoma",
        "mutación",
    ]

    for i, concept in enumerate(concepts):
        if i % 4 == 0:
            print()
        print(f"{concept:20}", end="")
    print()


def main():
    """Main function"""
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = "quickstart"

    if mode == "quickstart":
        show_quickstart()
    elif mode == "detailed":
        show_detailed_usage()
    elif mode == "troubleshooting":
        show_troubleshooting()
    elif mode == "examples":
        show_examples()
    elif mode == "all":
        show_quickstart()
        show_detailed_usage()
        show_troubleshooting()
        show_examples()
    else:
        print(
            "Uso: python usage_guide.py [quickstart|detailed|troubleshooting|examples|all]"
        )
        sys.exit(1)

    print_header("🎯 ¡Listo para usar BioLearn-GPT!")
    print(
        """
Para comenzar:
1. python setup.py (si no lo has hecho)
2. python src/main.py --mode all
3. python src/gradio_app.py

¡Disfruta generando contenido educativo con IA! 🧬✨
"""
    )


if __name__ == "__main__":
    main()
