from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return transforms.functional.pad(image, padding, 0, 'constant')

transform = transforms.Compose([
    SquarePad(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_model(num_classes):
    model = models.mobilenet_v2(weights=None) 
    
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )
    return model

try:
    with open("classes.txt", "r") as f:
        class_names = f.read().split(",")
    
    print(f"Loading Model for {len(class_names)} classes...")
    
    model = get_model(len(class_names))
    model.load_state_dict(torch.load("model.pth", map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    print("MobileNetV2 Model Loaded Successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("GUI Connected")
    try:
        while True:
            data = await websocket.receive_text()
            
            if "base64," in data:
                data = data.split("base64,")[1]
            
            img_bytes = base64.b64decode(data)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            
            img_t = transform(img).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                output = model(img_t)
                probs = torch.nn.functional.softmax(output[0], dim=0)
                score, idx = torch.max(probs, 0)
            
            label = class_names[idx.item()]
            conf = score.item()
            
            await websocket.send_json({
                "label": label,
                "confidence": f"{conf*100:.1f}%"
            })
            
    except Exception as e:
        print(f"Disconnected: {e}")