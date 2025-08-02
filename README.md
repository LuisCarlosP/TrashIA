Use Python 3.11.9

python -m venv venv

.\venv\Scripts\activate

Download 

pip install -r requirements.txt

Run

uvicorn app:app --reload

.env
HOST=0.0.0.0
PORT=8000
MODEL_PATH=models/modelo_basura.h5
ALLOWED_ORIGINS=http://localhost:8080
