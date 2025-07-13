import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.xgboost
import os

# Hent MLflow tracking URI fra environment variabel, fallback til /app/mlruns i container
mlruns_path = os.environ.get("MLFLOW_TRACKING_URI", "file:///app/mlruns")
mlflow.set_tracking_uri(mlruns_path)
mlflow.set_experiment("xgboost_creditcard")

# Indlæs data - creditcard.csv skal ligge i /app i containeren
data_path = "creditcard.csv"
df = pd.read_csv(data_path)

# Split features og target
X = df.drop("Class", axis=1)
y = df["Class"]

# Split til train og test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow run
with mlflow.start_run():
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 5,
        "eta": 0.1,
        "random_state": 42
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    mlflow.log_params(params)
    mlflow.log_metric("accuracy", acc)

    input_example = X_test.iloc[:1]
    mlflow.xgboost.log_model(model, "model", input_example=input_example)

    print(f"Accuracy: {acc:.4f}")

