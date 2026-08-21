from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

app = FastAPI()

GCS_BUCKET = os.environ.get("GCS_BUCKET", "")
S3_BUCKET = os.environ.get("S3_BUCKET", os.environ.get("CLOUD_BUCKET", ""))
MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")


def download_model():
    """
    Tai file model.pkl tu Cloud Storage (AWS S3 / GCP GCS) ve may khi server khoi dong.
    """
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # 1. Thu tai tu AWS S3 qua boto3
    if os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("AWS_DEFAULT_REGION") or S3_BUCKET:
        try:
            import boto3
            s3 = boto3.client("s3")
            s3.download_file(S3_BUCKET, MODEL_KEY, MODEL_PATH)
            print(f"Model da duoc tai xuong tu AWS S3 (s3://{S3_BUCKET}/{MODEL_KEY}).")
            return
        except Exception as e:
            print(f"Khong tai duoc tu S3 ({e}), thu phuong thuc khac...")

    # 2. Thu tai tu Google Cloud Storage
    if GCS_BUCKET:
        try:
            from google.cloud import storage
            client = storage.Client()
            bucket = client.bucket(GCS_BUCKET)
            blob = bucket.blob(MODEL_KEY)
            blob.download_to_filename(MODEL_PATH)
            print(f"Model da duoc tai xuong tu GCS (gs://{GCS_BUCKET}/{MODEL_KEY}).")
            return
        except Exception as e:
            print(f"Khong tai duoc tu GCS ({e}).")

    print("Chua cau hinh Cloud bucket hoac khong tai duoc, su dung model cuc bo neu co.")


download_model()

# Load model tu MODEL_PATH hoac fallback models/model.pkl neu MODEL_PATH chua co
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
elif os.path.exists("models/model.pkl"):
    model = joblib.load("models/model.pkl")
else:
    model = None


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    """
    Endpoint kiem tra suc khoe server.
    GitHub Actions goi endpoint nay sau khi deploy de xac nhan server dang chay.

    Tra ve: {"status": "ok"}
    """
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Endpoint suy luan chinh.

    Dau vao : JSON {"features": [f1, f2, ..., f12]}
    Dau ra  : JSON {"prediction": <0|1|2>, "label": <"thap"|"trung_binh"|"cao">}

    Thu tu 12 dac trung (khop voi thu tu trong FEATURE_NAMES cua test):
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
        chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
        pH, sulphates, alcohol, wine_type
    """
    if len(req.features) != 12:
        raise HTTPException(
            status_code=400,
            detail="Expected 12 features (wine quality)"
        )

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded"
        )

    pred = int(model.predict([req.features])[0])
    label_map = {0: "thap", 1: "trung_binh", 2: "cao"}
    label = label_map.get(pred, "unknown")

    return {"prediction": pred, "label": label}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
