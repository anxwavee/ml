# Iris ML API. FastAPI and Docker

Logistic Regression model 
trained on the Iris dataset
served with FastAPI

## Run locally
pip3 install -r requirements.txt
python3 train.py
uvicorn main:app --reload

## Run with Docker
docker build -t ml-fastapi-app .
docker run -d -p 8000:8000 ml-fastapi-app

## Endpoints
- GET / — health check
- POST /predict — input: sepal_length, sepal_width, petal_length, petal_width
- /docs — Swagger UI