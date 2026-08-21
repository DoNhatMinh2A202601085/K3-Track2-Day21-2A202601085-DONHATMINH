import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

EVAL_THRESHOLD = 0.70

# Bonus 1: Ho tro MLflow Tracking URI tu bien moi truong (DagsHub hoac SQLite cuc bo)
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(tracking_uri)


def check_data_drift(y_train: pd.Series) -> dict:
    """
    Bonus 5: Kiem tra phan phoi nhan va canh bao lech lac du lieu (Data Imbalance / Drift).
    """
    total_samples = len(y_train)
    dist = y_train.value_counts(normalize=True).to_dict()
    
    print("\n" + "="*50)
    print("BONUS 5: KIEM TRA PHAN PHOI NHAN (DATA DISTRIBUTION)")
    print("="*50)
    
    for cls in [0, 1, 2]:
        pct = dist.get(cls, 0.0) * 100
        print(f"  - Lop {cls}: {pct:.2f}% tong so mau ({int(pct * total_samples / 100)}/{total_samples})")
        if pct < 10.0:
            print(f"  [CANH BAO] Lop {cls} chiem it hon 10% ({pct:.2f}%). Du lieu co hien tuong lech nhan!")
            
    print("="*50 + "\n")
    return {str(k): round(float(v), 4) for k, v in dist.items()}


def get_model(model_type: str, params: dict):
    """
    Bonus 2: Khoi tao mo hinh dua tren model_type trong params.yaml.
    """
    # Loc bo cac tham so khong phu hop theo tung thuat toan
    p = params.copy()
    
    if model_type == "gradient_boosting":
        valid_params = {k: v for k, v in p.items() if k in ["n_estimators", "max_depth", "min_samples_split", "learning_rate"]}
        return GradientBoostingClassifier(**valid_params, random_state=42)
    elif model_type == "extra_trees":
        valid_params = {k: v for k, v in p.items() if k in ["n_estimators", "max_depth", "min_samples_split"]}
        return ExtraTreesClassifier(**valid_params, random_state=42)
    elif model_type == "logistic_regression":
        valid_params = {k: v for k, v in p.items() if k in ["C", "penalty", "solver"]}
        return LogisticRegression(**valid_params, max_iter=1000, random_state=42)
    else:
        # Default: RandomForestClassifier
        valid_params = {k: v for k, v in p.items() if k in ["n_estimators", "max_depth", "min_samples_split", "criterion"]}
        return RandomForestClassifier(**valid_params, random_state=42)


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.
    """
    # 1. Doc du lieu
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # 2. Tach dac trung (X) va nhan (y)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # Bonus 5: Kiem tra phan phoi du lieu
    class_dist = check_data_drift(y_train)

    # Bonus 2: Xac dinh thuat toan (model_type)
    model_params = params.copy()
    model_type = model_params.pop("model_type", "random_forest")

    with mlflow.start_run():
        # Ghi nhan tham so va model_type vao MLflow
        mlflow.log_param("model_type", model_type)
        mlflow.log_params(model_params)

        # Khoi tao va fit mo hinh
        model = get_model(model_type, model_params)
        model.fit(X_train, y_train)

        # Du doan va tinh toan metrics
        preds = model.predict(X_eval)
        acc = float(accuracy_score(y_eval, preds))
        f1 = float(f1_score(y_eval, preds, average="weighted"))

        # Log metrics vao MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        for cls_name, ratio in class_dist.items():
            mlflow.log_metric(f"class_{cls_name}_ratio", ratio)

        mlflow.sklearn.log_model(model, "model")

        # Bonus 3: Tao bao cao hieu suat chi tiet (Precision, Recall, Confusion Matrix)
        conf_mat = confusion_matrix(y_eval, preds)
        cls_report = classification_report(y_eval, preds, digits=4)
        
        report_text = "="*60 + "\n"
        report_text += f"BAO CAO HIEU SUAT MO HINH: {model_type.upper()}\n"
        report_text += "="*60 + "\n\n"
        report_text += f"Do chinh xac (Accuracy) : {acc:.4f}\n"
        report_text += f"F1 Score (Weighted)    : {f1:.4f}\n\n"
        report_text += "--- CONFUSION MATRIX ---\n"
        report_text += str(conf_mat) + "\n\n"
        report_text += "--- CLASSIFICATION REPORT (Precision / Recall theo lop) ---\n"
        report_text += cls_report + "\n"
        report_text += "="*60 + "\n"

        os.makedirs("outputs", exist_ok=True)
        with open("outputs/report.txt", "w", encoding="utf-8") as f:
            f.write(report_text)

        print(report_text)

        # Luu metrics.json (kem thong tin phan phoi nhan Bonus 5)
        metrics_data = {
            "model_type": model_type,
            "accuracy": acc,
            "f1_score": f1,
            "class_distribution": class_dist
        }
        with open("outputs/metrics.json", "w") as f:
            json.dump(metrics_data, f, indent=2)

        # Luu model.pkl
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
