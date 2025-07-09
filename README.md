# BioLearn-GPT: Generador de Contenido Educativo en Bioingeniería

## 🧬 Descripción del Proyecto

BioLearn-GPT es un sistema avanzado de generación de contenido educativo basado en inteligencia artificial, específicamente diseñado para crear explicaciones adaptativas sobre conceptos de bioingeniería y fisiología. El proyecto utiliza técnicas de fine-tuning de modelos de lenguaje (GPT-2) para generar explicaciones personalizadas según el nivel de conocimiento del usuario.

### 🎯 Objetivos

- **Objetivo General**: Desarrollar un sistema basado en modelos Transformer capaz de generar explicaciones coherentes y factualmente precisas sobre conceptos complejos de bioingeniería y fisiología, adaptando el nivel de detalle al conocimiento previo del usuario.

- **Objetivos Específicos**:
  1. Recopilar y curar un dataset de textos educativos sobre bioingeniería
  2. Implementar un sistema de etiquetado por niveles de dificultad
  3. Realizar fine-tuning de un modelo GPT-2 en español
  4. Desarrollar un mecanismo de prompting condicional
  5. Evaluar el modelo cuantitativa y cualitativamente

## 🚀 Características Principales

- **Generación Adaptativa**: Produce explicaciones en tres niveles de dificultad (Principiante, Intermedio, Experto)
- **Dominio Específico**: Especializado en bioingeniería y fisiología
- **Interfaz Web**: Interfaz intuitiva desarrollada con Gradio
- **Pipeline Completo**: Desde scraping de datos hasta evaluación del modelo
- **Evaluación Exhaustiva**: Métricas cuantitativas y cualitativas

## 📁 Estructura del Proyecto

```
BioGPT/
├── src/
│   ├── main.py                 # Pipeline principal
│   ├── config.py              # Configuraciones del proyecto
│   ├── web_scraper.py         # Scraping de datos educativos
│   ├── data_processor.py      # Procesamiento y limpieza de datos
│   ├── model_trainer.py       # Entrenamiento del modelo
│   ├── model_evaluator.py     # Evaluación del modelo
│   └── gradio_app.py          # Interfaz web
├── data/
│   ├── scraped_data/          # Datos recopilados
│   └── processed/             # Datos procesados para entrenamiento
├── biogpt_model/              # Modelo entrenado
├── requirements.txt           # Dependencias
└── README.md                  # Este archivo
```

## 🛠️ Instalación y Configuración

### Requisitos Previos

- Python 3.8+
- GPU recomendada (opcional, pero acelera el entrenamiento)
- 8GB+ de RAM

### Instalación

1. **Clonar el repositorio**:

```bash
git clone <repository-url>
cd BioGPT
```

2. **Crear entorno virtual**:

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias**:

```bash
pip install -r requirements.txt
```

4. **Configurar NLTK** (si es necesario):

```bash
python -c "import nltk; nltk.download('punkt')"
```

## 📊 Uso del Sistema

### 1. Ejecución del Pipeline Completo

Para ejecutar todo el pipeline desde cero:

```bash
cd src
python main.py --mode all
```

### 2. Ejecución por Fases

**Scraping de datos**:

```bash
python main.py --mode scrape
```

**Procesamiento de datos**:

```bash
python main.py --mode process
```

**Entrenamiento del modelo**:

```bash
python main.py --mode train
```

**Evaluación del modelo**:

```bash
python main.py --mode evaluate
```

### 3. Interfaz Web

Lanzar la interfaz Gradio:

```bash
python gradio_app.py
```

Acceder a: `http://localhost:7860`

## 🔧 Configuración Avanzada

### Parámetros del Modelo

Editar `config.py` para ajustar:

```python
@dataclass
class ModelConfig:
    base_model_name: str = "DeepESP/gpt2-spanish"
    learning_rate: float = 5e-5
    batch_size: int = 4
    num_epochs: int = 3
    max_length: int = 512
    # ... más parámetros
```

### Configuración de Scraping

```python
@dataclass
class ScrapingConfig:
    max_pages_per_source: int = 100
    delay_between_requests: float = 1.0
    timeout: int = 10
    # ... más configuraciones
```

## 📈 Evaluación y Métricas

El sistema incluye evaluación exhaustiva:

### Métricas Cuantitativas

- **Perplejidad**: Mide la calidad del modelo de lenguaje
- **Coherencia**: Análisis de fluidez textual
- **Relevancia**: Pertinencia del contenido generado
- **Completitud**: Evaluación de la exhaustividad

### Métricas Cualitativas

- **Consistencia de Nivel**: Evaluación de la adaptación al nivel solicitado
- **Precisión Factual**: Verificación de la exactitud científica
- **Adecuación Educativa**: Evaluación pedagógica del contenido

### Resultados de Ejemplo

```
Concept: célula (Principiante)
- Avg sentence length: 12.3
- Technical terms: 2
- Estimated level: principiante
- Coherence score: 0.85
```

## 🔍 Ejemplos de Uso

### Nivel Principiante

```
Prompt: "[PRINCIPIANTE] Explica qué es una célula"
Salida: "La célula es la unidad más pequeña de la vida. Todos los seres vivos están formados por células..."
```

### Nivel Intermedio

```
Prompt: "[INTERMEDIO] Explica la homeostasis"
Salida: "La homeostasis es el proceso mediante el cual los organismos mantienen un equilibrio interno estable..."
```

### Nivel Experto

```
Prompt: "[EXPERTO] Explica la transducción de señales"
Salida: "La transducción de señales celulares implica cascadas de fosforilación que regulan la expresión génica..."
```

## 🚨 Limitaciones y Consideraciones

### Limitaciones Conocidas

1. **Alucinaciones**: El modelo puede generar información incorrecta
2. **Datos Limitados**: Dependiente de la calidad del dataset de entrenamiento
3. **Evaluación Subjetiva**: La calidad educativa es difícil de medir objetivamente

### Mitigaciones

- Curación cuidadosa del dataset
- Evaluación humana complementaria
- Advertencias sobre verificación de contenido
