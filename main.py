from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
import torch
from PIL import Image
import io
import numpy as np
import cv2
import uuid
import shutil
from pathlib import Path
from app.model import BrainTumorCNN
from app.utils import get_transform
from app.visualize import GradCAM
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ This allows ALL origins. Use specific URLs in production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Create temp directory for image storage
TEMP_DIR = Path("temp_images")
TEMP_DIR.mkdir(exist_ok=True)

@app.on_event("shutdown")
async def cleanup_temp_files():
    """Clean up the temporary image directory when shutting down the app"""
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)

# Modeli yükləyin
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
model = BrainTumorCNN(num_classes=len(class_names))
model_path = 'model/brain_tumor_classification_model.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Generate Grad-CAM and save images to temp folder
def generate_gradcam(model, input_tensor, pred_idx):
    # Generate unique session ID for this prediction
    session_id = str(uuid.uuid4())
    session_dir = TEMP_DIR / session_id
    session_dir.mkdir(exist_ok=True)
    
    # Generate Grad-CAM
    grad_cam = GradCAM(model, model.conv3)
    cam = grad_cam(input_tensor, pred_idx)

    # Resize CAM
    cam_resized = np.uint8(255 * cam)
    cam_resized = np.repeat(cam_resized[:, :, np.newaxis], 3, axis=2)
    cam_resized = cv2.resize(cam_resized, (150, 150))

    # Create heatmap
    heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # De-normalize image - detach the tensor first
    img = input_tensor.detach().cpu().numpy()[0].transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    img = np.uint8(255 * img)

    # Create overlay
    alpha = 0.4
    overlay = heatmap * alpha + img * (1 - alpha)
    overlay = overlay.astype(np.uint8)

    # Save images to files
    original_path = session_dir / "original.png"
    heatmap_path = session_dir / "heatmap.png"
    overlay_path = session_dir / "overlay.png"
    
    cv2.imwrite(str(original_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(heatmap_path), cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    return {
        "session_id": session_id,
        "original": str(original_path),
        "heatmap": str(heatmap_path),
        "overlay": str(overlay_path)
    }

# Return a specific visualization image
@app.get("/visualizations/{session_id}/{image_type}")
async def get_visualization(session_id: str, image_type: str):
    valid_types = ["original", "heatmap", "overlay"]
    if image_type not in valid_types:
        return JSONResponse(
            content={"error": f"Invalid image type. Must be one of: {', '.join(valid_types)}"},
            status_code=400
        )
    
    image_path = TEMP_DIR / session_id / f"{image_type}.png"
    if not image_path.exists():
        return JSONResponse(
            content={"error": "Image not found"},
            status_code=404
        )
    
    return FileResponse(image_path)

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

        # **Generate user-friendly label and treatment recommendations**
        tumor_labels = {
            "glioma_tumor": "Glioma Şişi",
            "meningioma_tumor": "Meningioma Şişi",
            "no_tumor": "Şiş Tapılmadı",
            "pituitary_tumor": "Hipofiz Şişi",
        }

        treatment_recommendations = {
            "glioma_tumor": "Onkoloq və neyrocərrahla məsləhətləşin. Mümkün müalicə: əməliyyat, radioterapiya.",
            "meningioma_tumor": "Neyrocərrahla görüşün. Əməliyyat və ya müşahidə tələb oluna bilər.",
            "no_tumor": "Hər hansı patoloji tapılmadı. Yenidən müayinədən keçməyiniz tövsiyə olunur.",
            "pituitary_tumor": "Endokrinoloq və neyrocərrah məsləhəti alın. Hormonal terapiya və ya əməliyyat tələb oluna bilər.",
        }

        label = tumor_labels.get(predicted_class, "Naməlum Şiş")
        treatment = treatment_recommendations.get(predicted_class, "Müalicə tövsiyə edilmir.")

        input_tensor_gradcam = transform(image).unsqueeze(0).to(device)
        input_tensor_gradcam.requires_grad_(True)
        visualization = generate_gradcam(model, input_tensor_gradcam, pred.item())

        base_url = f"/visualizations/{visualization['session_id']}"
        image_urls = {
            "original_url": f"{base_url}/original",
            "heatmap_url": f"{base_url}/heatmap",
            "overlay_url": f"{base_url}/overlay"
        }

        # JSON cavabını qaytarın
        return JSONResponse(content={
            "prediction": label,
            "confidence": confidence,
            "treatment_recommendation": treatment,
            "visualization_urls": image_urls
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
