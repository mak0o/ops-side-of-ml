import io
import json
import urllib.request

import mlflow
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import models, transforms

app = FastAPI()

Image.MAX_IMAGE_PIXELS = 50_000_000
MAX_UPLOAD = 10 * 1024 * 1024  # 10MB

# 1. 学習済みモデルの読み込み (ResNet50)
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.eval()

# 2. 画像の前処理定義
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 3. ImageNetラベルの取得 (起動時に一度だけ実行)
# 毎回ダウンロードするのは無駄なので、グローバル変数に保持します
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
try:
    with urllib.request.urlopen(LABELS_URL) as url:
        imagenet_labels = json.loads(url.read().decode())
        print("ImageNet labels loaded successfully.")
except Exception as e:
    print(f"Failed to load labels: {e}")
    imagenet_labels = []  # 失敗時は空リストにしてエラーを防ぐ

# MLflowの設定
try:
    mlflow.set_tracking_uri("http://mlflow-server:5000")
    mlflow.set_experiment("resnet-predictions")
    MLFLOW_READY = True
except Exception as e:
    print(f"MLflow unavailable: {e}")
    MLFLOW_READY = False


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 画像データの読み込み
    image_data = await file.read(MAX_UPLOAD + 1)
    if len(image_data) > MAX_UPLOAD:
        raise HTTPException(413, "file too large")
    try:
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "invalid image") from None
    

    # 前処理と推論
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)

    # 結果の取得 (Softmaxで確率に変換)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    confidence, predicted_idx = torch.max(probabilities, 0)

    prediction_id = int(predicted_idx.item())
    confidence_score = float(confidence.item())

    # IDからラベル名を取得
    # リストのインデックスがIDに対応しています
    predicted_label = imagenet_labels[prediction_id] if imagenet_labels else "Unknown"

    # MLflowへのログ記録
    if MLFLOW_READY:
        try:
            with mlflow.start_run():
                mlflow.log_param("image_name", file.filename)
                mlflow.log_metric("confidence", confidence_score)
                mlflow.log_param("predicted_id", prediction_id)
                mlflow.log_param("predicted_label", predicted_label)
        except Exception as e:
            print(f"MLflow logging failed: {e}")

    return {
        "top_prediction_id": prediction_id,
        "predicted_label": predicted_label,  # JSONレスポンスにも追加
        "confidence": confidence_score
    }