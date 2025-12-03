# TrashIA - AI Trash Classifier

REST API built with FastAPI and TensorFlow to classify types of trash, determine recyclability, and provide AI-powered chat assistance.

## Features

- Image classification into 6 categories: cardboard, glass, metal, paper, plastic, and general trash
- Automatic recyclability determination
- Interactive AI chat (Google Gemini) for recycling queries
- Rate limiting for API protection
- Multi-language support (English/Spanish)
- File validation by MIME type
- Automatic documentation with Swagger/OpenAPI

## Requirements

- Python 3.11.9
- pip
- Google Gemini API key (optional, for chat functionality)

## Local Installation

### 1. Clone the repository
```bash
git clone https://github.com/LuisCarlosP/TrashIA.git
cd TrashIA
```

### 2. Create virtual environment
```bash
python -m venv venv
```

### 3. Activate virtual environment
```bash
# Windows
.\venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 4. Install dependencies
```bash
cd ModeloIATrashNet
pip install -r requirements.txt
```

### 5. Configure environment variables
Create a `.env` file in `ModeloIATrashNet/`:
```env
HOST=0.0.0.0
PORT=8000
MODEL_PATH=models/modelo_basura.h5
ALLOWED_ORIGINS=http://localhost:8080,https://luiscarlosp.github.io
GEMINI_API_KEY=use_your_gemini_api_key
```

### 6. Run the application
```bash
python run.py
```

The API will be available at: `http://localhost:8000`

## Endpoints

### General
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information and available endpoints |
| GET | `/health` | Check API status |
| GET | `/docs` | Interactive Swagger UI documentation |

### Prediction
| Method | Endpoint | Description | Rate Limit |
|--------|----------|-------------|------------|
| POST | `/predict` | Classify trash image | 10/minute |

**Parameters for `/predict`:**
- `file`: Image (JPEG, PNG) - Maximum 5MB

**Successful response:**
```json
{
  "material_type": "plastic",
  "is_recyclable": true,
  "confidence": 0.95,
  "material_info": "Information about the material"
}
```

### Chat
| Method | Endpoint | Description | Rate Limit |
|--------|----------|-------------|------------|
| POST | `/chat/session` | Create chat session | 20/minute |
| POST | `/chat/message` | Send message to chat | 30/minute |
| GET | `/chat/history/{session_id}` | Get conversation history | 20/minute |
| PUT | `/chat/material` | Update material context | 20/minute |
| DELETE | `/chat/session/{session_id}` | Delete session | 10/minute |

**Parameters for `/chat/session`:**
- `material_type`: (optional) Identified material type
- `is_recyclable`: (optional) Whether the material is recyclable
- `material_info`: (optional) Additional material information
- `language`: Chat language (`en`/`es`, default: `en`)

**Parameters for `/chat/message`:**
- `session_id`: Chat session ID
- `message`: User message

### Location
| Method | Endpoint | Description | Rate Limit |
|--------|----------|-------------|------------|
| GET | `/location/recycling-points` | Get nearby recycling points | 30/minute |
| POST | `/location/recycling-points/search` | Search recycling points (POST) | 30/minute |
| GET | `/location/health` | Location service health check | - |

**Parameters for `/location/recycling-points`:**
- `latitude`: Latitude (-90 to 90)
- `longitude`: Longitude (-180 to 180)
- `radius`: Search radius in meters (100 to 50000, default: 5000)
- `types`: (optional) Filter by recycling point types

## Classification Categories

| Category | Recyclable | Description |
|----------|------------|-------------|
| cardboard | Yes | Cardboard and boxes |
| glass | Yes | Glass and crystal |
| metal | Yes | Metals and cans |
| paper | Yes | Paper and documents |
| plastic | Yes | Plastics |
| trash | No | General non-recyclable waste |

## Project Structure

```
ModeloIATrashNet/
├── app.py                 # Main FastAPI application
├── run.py                 # Startup script
├── requirements.txt       # Python dependencies
├── config/
│   ├── settings.py        # Configuration and environment variables
│   ├── chat_prompts.json  # AI chat prompts
│   └── recyclable_info.json # Material information
├── core/
│   └── dependencies.py    # Dependency injection
├── exceptions/
│   ├── image_exceptions.py
│   ├── model_exceptions.py
│   └── validation_exceptions.py
├── models/
│   ├── modelo_basura.h5   # Alternative TensorFlow model
│   └── TrashIAv2.h5       # Main model
├── routes/
│   ├── prediction.py      # Prediction routes
│   ├── chat.py            # Chat routes
│   └── location.py        # Location/recycling points routes
├── services/
│   ├── trash_services.py  # Classification logic
│   ├── chat_service.py    # Gemini chat logic
│   └── location_service.py # OpenStreetMap integration
└── scripts/
    └── test_model.py      # Test scripts
```

## Technologies

| Technology | Version | Usage |
|------------|---------|-------|
| FastAPI | 0.116.1 | Web framework |
| TensorFlow | 2.19.0 | Classification model |
| Keras | 3.10.0 | Neural network API |
| Uvicorn | 0.35.0 | ASGI server |
| Pillow | 11.3.0 | Image processing |
| Pydantic | 2.11.7 | Data validation |
| Google Generative AI | 0.8.3 | Gemini chat |
| SlowAPI | 0.1.9 | Rate limiting |
| python-magic | 0.4.27 | MIME validation |
| httpx | 0.28.1 | HTTP client (OpenStreetMap) |
| python-dotenv | 1.1.1 | Environment variables |

## License

This project is open source.

