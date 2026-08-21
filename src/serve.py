from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
import joblib
import os

app = FastAPI()

ARTIFACT_BUCKET = os.environ["ARTIFACT_BUCKET"]
MODEL_KEY = "artifacts/current/model.joblib"
MODEL_PATH = os.path.expanduser("~/models/model.joblib")

# Nhan tra ve cho tung gia tri du doan cua mo hinh
LABELS = {0: "thu_nhap_thap", 1: "thu_nhap_cao"}

# So dac trung dau vao bat buoc (khop voi FEATURE_NAMES trong tests/test_train.py)
N_FEATURES = 10


def download_model():
    """
    Tai file model.joblib tu S3 ve may khi server khoi dong.

    Ham nay duoc goi mot lan khi module duoc import. boto3 tu tim credentials theo
    thu tu: bien moi truong -> ~/.aws/credentials -> IAM role gan vao EC2 instance.
    """
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # TODO 1: Tao S3 client
    s3 = boto3.client("s3")

    # TODO 2 + 3: Tai file model xuong may
    s3.download_file(ARTIFACT_BUCKET, MODEL_KEY, MODEL_PATH)

    # TODO 4: In thong bao thanh cong
    print(f"Model da duoc tai xuong tu s3://{ARTIFACT_BUCKET}/{MODEL_KEY}")


download_model()
model = joblib.load(MODEL_PATH)


class ScoreRequest(BaseModel):
    features: list[float]


@app.get("/healthz")
def healthz():
    """
    Endpoint kiem tra suc khoe server.
    GitHub Actions goi endpoint nay sau khi deploy de xac nhan server dang chay.

    Tra ve: {"status": "ok"}
    """
    # TODO 5: Tra ve dict {"status": "ok"}
    return {"status": "ok"}


@app.post("/score")
def score(req: ScoreRequest):
    """
    Endpoint suy luan chinh.

    Dau vao : JSON {"features": [f1, f2, ..., f10]}
    Dau ra  : JSON {"prediction": <0|1>, "label": <"thu_nhap_thap"|"thu_nhap_cao">}

    Thu tu 10 dac trung (khop voi thu tu trong FEATURE_NAMES cua test):
        age, workclass, education_num, marital_status, occupation,
        relationship, sex, capital_gain, capital_loss, hours_per_week
    """
    # TODO 6: Kiem tra so luong dac trung.
    if len(req.features) != N_FEATURES:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {N_FEATURES} features (adult income), got {len(req.features)}",
        )

    # TODO 7: Goi model.predict([req.features]) de lay ket qua du doan.
    pred = int(model.predict([req.features])[0])

    # TODO 8: Tra ve dict chua "prediction" (int) va "label" (string).
    return {"prediction": pred, "label": LABELS[pred]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
