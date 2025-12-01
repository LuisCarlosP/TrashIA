# TrashIA - Clasificador de Basura con IA

API de FastAPI con TensorFlow para clasificar tipos de basura y determinar reciclabilidad.

## Requisitos

- Python 3.11.9
- pip

## Instalación Local

### 1. Crear entorno virtual
```bash
python -m venv venv
```

### 2. Activar entorno virtual
```bash
# Windows
.\venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno
Crea un archivo `.env` en `ModeloIATrashNet/`:
```env
HOST=0.0.0.0
PORT=8000
MODEL_PATH=models/modelo_basura.h5
ALLOWED_ORIGINS=http://localhost:8080
```

### 5. Ejecutar la aplicación
```bash
cd ModeloIATrashNet
python run.py
```

La API estará disponible en: `http://localhost:8000`

### Endpoints disponibles:
- `GET /health` - Verificar estado de la API
- `POST /predict` - Clasificar imagen de basura

## Estructura del Proyecto

```
ModeloIATrashNet/
├── app.py                 # Aplicación FastAPI
├── run.py                 # Script de inicio
├── requirements.txt       # Dependencias
├── Dockerfile            # Configuración Docker
├── render.yaml           # Configuración Render
├── .dockerignore         # Exclusiones Docker
├── config/               # Configuración
├── core/                 # Dependencias centrales
├── exceptions/           # Excepciones personalizadas
├── models/               # Modelo ML (.h5)
├── routes/               # Rutas API
├── services/             # Lógica de negocio
└── scripts/              # Scripts de prueba
```

## Tecnologías

- **FastAPI** - Framework web
- **TensorFlow/Keras** - Modelo de clasificación
- **Uvicorn** - Servidor ASGI
- **Pillow** - Procesamiento de imágenes
- **Pydantic** - Validación de datos

