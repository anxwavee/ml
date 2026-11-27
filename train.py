import mlflow
import mlflow.sklearn
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("iris_experiment")

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

best_f1 = 0
best_run_id = None
best_model = None

for c in [0.01, 0.1, 1, 10]:
    with mlflow.start_run():
        model = LogisticRegression(C=c, max_iter=200)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average="macro")

        mlflow.log_param("C", c)
        mlflow.log_param("max_iter", 200)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1", f1)
        mlflow.sklearn.log_model(model, "model")

        print("C =", c, "accuracy =", acc, "f1 =", f1)

        if f1 > best_f1:
            best_f1 = f1
            best_run_id = mlflow.active_run().info.run_id
            best_model = model


joblib.dump(best_model, "model.joblib")

with mlflow.start_run(run_id=best_run_id):
    mlflow.log_artifact("model.joblib")

mlflow.register_model("runs:/" + best_run_id + "/model", "iris_model")
print("best f1:", best_f1)