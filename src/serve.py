from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
import joblib
import os

app = FastAPI()

GCS_BUCKET = os.environ.get("GCS_BUCKET", "")
GCS_MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")


def download_model():
    """
    Tai file model.pkl tu GCS ve may khi server khoi dong.

    Ham nay duoc goi mot lan khi module duoc import. Su dung
    GOOGLE_APPLICATION_CREDENTIALS de xac thuc (duoc dat trong systemd service).
    """
    if GCS_BUCKET:
        try:
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            # TODO 1: Tao storage.Client()
            client = storage.Client()
            # TODO 2: Lay bucket va blob tuong ung
            bucket = client.bucket(GCS_BUCKET)
            blob = bucket.blob(GCS_MODEL_KEY)
            # TODO 3: Tai file model xuong may
            blob.download_to_filename(MODEL_PATH)
            # TODO 4: In thong bao thanh cong
            print("Model da duoc tai xuong tu GCS.")
        except Exception as e:
            print(f"Loi khi tai model tu GCS: {e}")
    else:
        print("GCS_BUCKET chua duoc dat, su dung model cuc bo neu co.")


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
    # TODO 5: Tra ve dict {"status": "ok"}
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
    # TODO 6: Kiem tra so luong dac trung.
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

    # TODO 7: Goi model.predict([req.features]) de lay ket qua du doan.
    pred = int(model.predict([req.features])[0])

    # TODO 8: Tra ve dict chua "prediction" (int) va "label" (string).
    # Nhan tuong ung: 0 -> "thap", 1 -> "trung_binh", 2 -> "cao"
    label_map = {0: "thap", 1: "trung_binh", 2: "cao"}
    label = label_map.get(pred, "unknown")

    return {"prediction": pred, "label": label}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
