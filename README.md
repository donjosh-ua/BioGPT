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
- **Diagnóstico GPU/CUDA**: Herramientas integradas para verificar compatibilidad hardware

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
│   ├── gradio_app.py          # Interfaz web
│   ├── gpu_diagnostics.py     # Diagnóstico GPU/CUDA
│   └── fix_cuda.py           # Corrector automático CUDA
├── data/
│   ├── scraped_data/          # Datos recopilados
│   └── processed/             # Datos procesados para entrenamiento
├── biogpt_model/              # Modelo entrenado
├── setup.py                   # Configuración inicial del proyecto
├── demo.py                    # Script de demostración
├── usage_guide.py             # Guía de uso interactiva
├── test_installation.py       # Verificador de instalación
├── requirements.txt           # Dependencias
└── README.md                  # Este archivo
```

## 🛠️ Instalación y Configuración

### Requisitos Previos

- Python 3.8+
- GPU NVIDIA recomendada (opcional, pero acelera significativamente el entrenamiento)
- 8GB+ de RAM
- CUDA 11.8+ (para soporte GPU)

### Instalación Rápida

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

4. **Ejecutar configuración automática**:

```bash
python setup.py
```

### 🔧 Diagnóstico y Configuración GPU

#### Verificar Compatibilidad GPU/CUDA

Antes de entrenar el modelo, es crucial verificar que tu sistema tenga configuración GPU correcta:

```bash
# Diagnóstico completo del sistema
python src/gpu_diagnostics.py
```

Este comando verificará:

- ✅ Versión de drivers NVIDIA
- ✅ Versión de CUDA toolkit instalada
- ✅ Compatibilidad PyTorch-CUDA
- ✅ Estado de GPU detectadas
- ⚠️ Problemas de compatibilidad

#### Solución Automática de Problemas CUDA

Si se detectan problemas de compatibilidad:

```bash
# Corrector automático de problemas CUDA
python src/fix_cuda.py
```

Este script:

1. 🔍 Diagnostica problemas específicos
2. 📦 Desinstala versiones incompatibles de PyTorch
3. 🚀 Instala la versión correcta según tu hardware
4. ✅ Verifica la instalación

#### Problemas Comunes y Soluciones

| Problema                  | Síntoma                     | Solución                      |
| ------------------------- | --------------------------- | ----------------------------- |
| **PyTorch sin CUDA**      | `CUDA available: False`     | `python src/fix_cuda.py`      |
| **CUDA version mismatch** | Error durante entrenamiento | Reinstalar PyTorch compatible |
| **Drivers antiguos**      | GPU no detectada            | Actualizar drivers NVIDIA     |
| **CUDA no instalado**     | `nvcc: command not found`   | Instalar CUDA toolkit         |

#### Instalación Manual de PyTorch

Si necesitas instalar PyTorch manualmente para tu versión de CUDA:

```bash
# Para CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Para CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Para CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# CPU only (sin GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 🧪 Verificación de Instalación

```bash
# Verificar que todo funciona correctamente
python test_installation.py
```

## 📊 Uso del Sistema

### 🚀 Inicio Rápido

```bash
# Ver guía interactiva de uso
python usage_guide.py

# Ejecutar demo del modelo
python demo.py

# Pipeline completo (primera ejecución)
python src/main.py --mode all
```

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
python src/gradio_app.py
```

Acceder a: `http://localhost:7860`

## ⚡ Optimización de Rendimiento

### Configuración para GPU

Si tienes GPU NVIDIA disponible, el entrenamiento será significativamente más rápido:

```python
# En config.py - Configuración optimizada para GPU
@dataclass
class ModelConfig:
    batch_size: int = 8              # Aumentar si tienes más VRAM
    gradient_accumulation_steps: int = 2  # Reducir con batch_size mayor
    num_epochs: int = 3
    learning_rate: float = 5e-5
```

### Configuración para CPU

Si solo tienes CPU disponible:

```python
# En config.py - Configuración optimizada para CPU
@dataclass
class ModelConfig:
    batch_size: int = 2              # Reducir para evitar memoria insuficiente
    gradient_accumulation_steps: int = 8  # Aumentar para simular batch mayor
    num_epochs: int = 1              # Reducir para tiempo de entrenamiento
    learning_rate: float = 3e-5      # Tasa más conservadora
```

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

## 🚨 Solución de Problemas

### Problemas de GPU/CUDA

```bash
# Diagnóstico completo
python src/gpu_diagnostics.py

# Solución automática
python src/fix_cuda.py

# Verificación post-instalación
python -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}')"
```

### Problemas de Memoria

```bash
# Reducir batch_size en config.py
# Aumentar gradient_accumulation_steps
# Usar modelo más pequeño si es necesario
```

### Problemas de Datos

```bash
# Verificar datos scraped
ls data/scraped_data/

# Reprocesar datos si es necesario
python src/main.py --mode process

# Verificar datos procesados
ls data/processed/
```

### Logs y Debugging

```bash
# Ejecutar con verbose logging
python src/main.py --mode all --verbose

# Verificar logs de entrenamiento
ls biogpt_model/logs/
```

## 🚨 Limitaciones y Consideraciones

### Limitaciones Conocidas

1. **Alucinaciones**: El modelo puede generar información incorrecta
2. **Datos Limitados**: Dependiente de la calidad del dataset de entrenamiento
3. **Evaluación Subjetiva**: La calidad educativa es difícil de medir objetivamente
4. **Requisitos de Hardware**: El entrenamiento óptimo requiere GPU NVIDIA

### Mitigaciones

- Curación cuidadosa del dataset
- Evaluación humana complementaria
- Advertencias sobre verificación de contenido
- Herramientas de diagnóstico GPU integradas
- Configuraciones optimizadas para diferentes tipos de hardware
