import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

EVAL_THRESHOLD = 0.70

# Thiết lập MLflow tracking URI mặc định nếu chưa có
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(tracking_uri)


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huấn luyện mô hình và ghi nhận kết quả vào MLflow.

    Tham số:
        params     : dict chứa các siêu tham số cho RandomForestClassifier.
        data_path  : đường dẫn đến file dữ liệu huấn luyện.
        eval_path  : đường dẫn đến file dữ liệu đánh giá.

    Trả về:
        accuracy (float): độ chính xác trên tập đánh giá.
    """

    # TODO 1: Đọc dữ liệu huấn luyện và đánh giá
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # TODO 2: Tách đặc trưng (X) và nhãn (y)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    with mlflow.start_run():
        # TODO 3: Ghi nhận các siêu tham số
        mlflow.log_params(params)

        # TODO 4: Khởi tạo và huấn luyện RandomForestClassifier
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # TODO 5: Dự đoán trên tập đánh giá và tính chỉ số
        preds = model.predict(X_eval)
        acc = float(accuracy_score(y_eval, preds))
        f1 = float(f1_score(y_eval, preds, average="weighted"))

        # TODO 6: Ghi nhận chỉ số vào MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        # TODO 7: In kết quả ra màn hình
        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        # TODO 8: Lưu metrics ra file outputs/metrics.json
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/metrics.json", "w") as f:
            json.dump({"accuracy": acc, "f1_score": f1}, f)

        # TODO 9: Lưu mô hình ra file models/model.pkl
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    # TODO 10: Trả về acc
    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
