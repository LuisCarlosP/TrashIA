# TrashIA - AI Trash Classifier

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/LuisCarlosP/TrashIA)

REST API built with FastAPI and TensorFlow to classify types of trash, determine recyclability, and provide AI-powered chat assistance. The backend follows SOLID principles with dependency injection and protocol-based abstractions.

## Features

- Image classification into 6 categories: cardboard, glass, metal, paper, plastic, and general trash
- Automatic recyclability determination
- Interactive AI chat (Google Gemini) for recycling queries
- Barcode scanning for product information and recyclability
- Recycling point location search using OpenStreetMap
- Rate limiting for API protection
- Multi-language support (English/Spanish)
- File validation by MIME type
- Automatic documentation with Swagger/OpenAPI
- Security: API Key authentication, Redis-based rate limiting, and Circuit Breakers
- Comprehensive test suite
- SOLID architecture with dependency injection and protocol-based abstractions


## Requirements

- Python 3.11.9
- pip
- Google Gemini API key (optional, for chat functionality)
- Redis (required for rate limiting)


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
cd TrashIABackend
pip install -r requirements.txt
```

### 5. Configure environment variables
Create a `.env` file in `TrashIABackend/`:
```env
# Server Configuration
HOST=0.0.0.0
PORT=8000
ENVIRONMENT=development

# Model Configuration
MODEL_PATH=models/TrashIAv2.h5

# Security
API_KEY=your_secret_api_key
REDIS_URL=redis://localhost:6379/0

# CORS
ALLOWED_ORIGINS=http://localhost:8080,https://luiscarlosp.github.io

# External API Keys
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.0-flash

# File Upload Limits
MAX_FILE_SIZE_MB=5
ALLOWED_MIME_TYPES=image/jpeg,image/png,image/jpg

# Rate Limiting (requests per minute)
RATE_LIMIT_PREDICT=10/minute
RATE_LIMIT_CHAT_SESSION=20/minute
RATE_LIMIT_CHAT_MESSAGE=30/minute
RATE_LIMIT_CHAT_HISTORY=20/minute
RATE_LIMIT_CHAT_DELETE=10/minute
RATE_LIMIT_CHAT_UPDATE=20/minute
RATE_LIMIT_LOCATION=30/minute
RATE_LIMIT_BARCODE=30/minute

# Circuit Breaker
CIRCUIT_BREAKER_FAIL_MAX=5
CIRCUIT_BREAKER_RESET_TIMEOUT=60

# HTTP Timeouts (seconds)
HTTP_TIMEOUT_LOCATION=30.0
HTTP_TIMEOUT_BARCODE=10.0
HTTP_TIMEOUT_HEALTH_CHECK=5.0
OVERPASS_QUERY_TIMEOUT=25

# Location Service
LOCATION_CACHE_TTL_MINUTES=30
LOCATION_DEFAULT_RADIUS=2000
LOCATION_MIN_RADIUS=100
LOCATION_MAX_RADIUS=50000

# Barcode Service
BARCODE_MIN_LENGTH=8

# External API URLs
OPEN_FOOD_FACTS_URL=https://world.openfoodfacts.org/api/v2/product
UPCITEMDB_URL=https://api.upcitemdb.com/prod/trial/lookup
OVERPASS_SERVERS=https://overpass-api.de/api/interpreter,https://overpass.kumi.systems/api/interpreter,https://maps.mail.ru/osm/tools/overpass/api/interpreter
```

> **Note:** All environment variables are required. The application will fail to start if any are missing.

### 6. Run the application
```bash
python run.py
```

The API will be available at: `http://localhost:8000`

## Running Tests

To run the test suite:

```bash
cd TrashIABackend
pytest tests/ -v
```

To run specific test files:

```bash
pytest tests/routes/test_prediction.py -v
pytest tests/integration/test_file_validation.py -v
```

## Endpoints

### General
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information and available endpoints |
| GET | `/docs` | Interactive Swagger UI documentation |

### Health
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Check API status |
| GET | `/health/model` | Check ML model availability and health |
| GET | `/health/dependencies` | Check all external dependencies |
| GET | `/health/gemini` | Check Gemini API status |
| GET | `/health/osm` | Check OpenStreetMap status |
| GET | `/health/openfoodfacts` | Check OpenFoodFacts status |

### Prediction
| Method | Endpoint | Description | Rate Limit |
|--------|----------|-------------|------------|
| POST | `/predict` | Classify trash image | 10/minute |

**Parameters for `/predict`:**
- `file`: Image (JPEG, PNG) - Maximum 5MB
- `language`: Response language (`en`/`es`, default: `en`)

**Successful response:**
```json
{
  "class": "plastic",
  "confidence": 0.95,
  "is_recyclable": true,
  "message": "Information about the material"
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

### Barcode
| Method | Endpoint | Description | Rate Limit |
|--------|----------|-------------|------------|
| GET | `/barcode/{barcode}` | Get product information by barcode | 30/minute |
| GET | `/barcode/health` | Barcode service health check | - |

**Parameters for `/barcode/{barcode}`:**
- `barcode`: Product barcode (minimum 8 digits)

**Successful response:**
```json
{
  "found": true,
  "barcode": "12345678",
  "name": "Product Name",
  "brand": "Brand Name",
  "source": "openfoodfacts",
  "recycling_info": [
    {
      "material": "Plastic",
      "recyclable": true,
      "bin": "Yellow Bin",
      "bin_type": "yellow",
      "tip": "Clean/Rinse the container"
    }
  ]
}
```

### Location
| Method | Endpoint | Description | Rate Limit |
|--------|----------|-------------|------------|
| GET | `/location/recycling-points` | Get nearby recycling points | 30/minute |
| POST | `/location/recycling-points/search` | Search recycling points (POST) | 30/minute |
| GET | `/location/health` | Location service health check | - |

**Parameters for `/location/recycling-points`:**
- `latitude`: Latitude (-90 to 90)
- `longitude`: Longitude (-180 to 180)
- `radius`: Search radius in meters (100 to 50000, default: 2000)
- `types`: (optional) Filter by recycling point types

### Security
All endpoints (except `/health` and `/docs`) require authentication via header:
- `X-API-Key`: Your secret API key


## Classification Categories

| Category | Recyclable | Description |
|----------|------------|-------------|
| cardboard | Yes | Cardboard and boxes |
| glass | Yes | Glass and crystal |
| metal | Yes | Metals and cans |
| paper | Yes | Paper and documents |
| plastic | Yes | Plastics |
| trash | No | General non-recyclable waste |

## Architecture

The backend follows SOLID principles:

| Principle | Implementation |
|-----------|----------------|
| Single Responsibility | Each service class has one responsibility (ChatService, BarcodeService, LocationService) |
| Open/Closed | Services accept provider lists, allowing new providers without code modification |
| Liskov Substitution | All providers are interchangeable (OpenFoodFactsProvider, UPCItemDBProvider) |
| Interface Segregation | Small, focused protocols (ChatProviderProtocol, BarcodeProviderProtocol) |
| Dependency Inversion | Services depend on abstractions (protocols), not concrete implementations |


## Project Structure

```
TrashIABackend/
├── app.py                 # Main FastAPI application
├── run.py                 # Startup script
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker configuration
├── render.yaml            # Render deployment configuration
├── config/
│   ├── settings.py        # Configuration and environment variables
│   ├── chat_prompts.json  # AI chat prompts
│   └── recyclable_info.json # Material information
├── core/
│   ├── dependencies.py    # Dependency injection
│   ├── file_validator.py  # File validation logic
│   ├── security.py        # Security and authentication
│   ├── error_handler.py   # Structured error responses
│   └── protocols/         # Protocol definitions (interfaces)
│       ├── ai.py          # AI provider protocol
│       ├── barcode.py     # Barcode provider protocol
│       ├── cache.py       # Cache protocol
│       ├── chat.py        # Chat provider and repository protocols
│       ├── http.py        # HTTP client protocol
│       ├── response.py    # Response builder protocol
│       └── validation.py  # Validation protocol
├── exceptions/
│   ├── base_exception.py  # Base exception class
│   ├── external_api_exceptions.py # External API exceptions
│   ├── image_exceptions.py
│   ├── location_exceptions.py
│   ├── model_exceptions.py
│   └── validation_exceptions.py
├── models/
│   ├── location_models.py  # Location data models
│   └── TrashIAv2.h5       # Main classification model
├── routes/
│   ├── prediction.py      # Prediction routes
│   ├── chat.py            # Chat routes
│   ├── location.py        # Location/recycling points routes
│   ├── barcode.py         # Barcode scanning routes
│   └── health.py          # Health check routes
├── services/
│   ├── trash_services.py  # Classification logic
│   ├── chat_service.py    # Gemini chat logic
│   ├── chat_session_repository.py # Session storage
│   ├── location_service.py # OpenStreetMap integration
│   ├── location_cache.py  # Location caching
│   ├── barcode_service.py # Barcode product lookup
│   ├── osm_parser.py      # OpenStreetMap response parser
│   └── providers/         # External API providers
│       ├── gemini_provider.py    # Gemini AI provider
│       └── barcode_providers.py  # OpenFoodFacts and UPCItemDB providers
└── tests/
    ├── conftest.py         # Test configuration
    ├── factories/          # Test data factories
    │   ├── barcode_factory.py
    │   ├── chat_factory.py
    │   ├── location_factory.py
    │   └── prediction_factory.py
    ├── integration/        # Integration tests
    │   ├── test_file_validation.py
    │   ├── test_rate_limiting.py
    │   └── test_external_apis.py
    ├── routes/             # Route tests
    │   ├── test_prediction.py
    │   ├── test_chat.py
    │   ├── test_location.py
    │   ├── test_barcode.py
    │   └── test_health.py
    └── services/           # Service tests
        ├── test_trash_services.py
        ├── test_chat_service.py
        └── test_barcode_service.py
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
| httpx | 0.28.1 | HTTP client |
| python-dotenv | 1.1.1 | Environment variables |
| Redis | 5.0.1 | Rate limiting storage |
| pybreaker | 1.0.1 | Circuit breakers |
| pytest | 9.0.1 | Testing framework |

## License

Copyright 2025 Luis Carlos Picado Rojas - All Rights Reserved

This project is available for viewing and educational purposes only. See the [LICENSE](LICENSE) file for details.

---

## Author

**Luis Carlos Picado Rojas**

- GitHub: [@LuisCarlosP](https://github.com/LuisCarlosP)
