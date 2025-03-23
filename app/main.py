from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import io
import numpy as np
import cv2
from model import BrainTumorCNN
from utils import get_transform
from visualize import GradCAM

# FastAPI tətbiqini yaradın
app = FastAPI()

# Modeli yükləyin
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
model = BrainTumorCNN(num_classes=len(class_names))
model_path = 'brain_tumor_classification_model.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Şəkil yükləmək və proqnoz almaq üçün marşrut
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Faylın şəkil olub-olmadığını yoxlayın
    if not file.content_type.startswith('image/'):
        return JSONResponse(content={"error": "Fayl şəkil deyil"}, status_code=400)

    try:
        # Yüklənmiş faylı oxuyun və emal edin
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        transform = get_transform()
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Proqnozu əldə edin
        with torch.no_grad():
            output = model(input_tensor)
            _, pred = torch.max(output, 1)
            predicted_class = class_names[pred.item()]
            confidence = torch.nn.functional.softmax(output, dim=1)[0, pred.item()].item()

        # JSON cavabını qaytarın
        return JSONResponse(content={
            "prediction": predicted_class,
            "confidence": confidence
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)