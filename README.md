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


## FOR SIS3

Added Streamlit and MLflow.

- MLflow: experiment "iris_experiment", logged C, accuracy, f1 and model
- best model registered as "iris_model" (2 versions)
- app.py: Streamlit page that sends data to the API and shows prediction

Run:
python3 train.py
python3 -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
python3 -m uvicorn main:app --reload
python3 -m streamlit run app.py

Screenshots in screenshots folder.