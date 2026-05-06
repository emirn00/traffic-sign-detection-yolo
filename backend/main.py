import io
import os
from typing import List

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
from ultralytics import YOLO

# ============================================================
# Yapılandırma
# ============================================================
app = FastAPI(
    title="Traffic Sign Detection API",
    description="YOLOv8 tabanlı trafik işareti tespit servisi",
    version="1.0.0"
)

# Frontend erişimi için CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Geliştirme aşamasında her yerden erişime izin ver
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model yükleme
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "best.pt")
try:
    model = YOLO(MODEL_PATH)
    print(f"✅ Model başarıyla yüklendi: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Model yüklenirken hata oluştu: {str(e)}")
    # Model bulunamazsa uygulama çöksün istemiyoruz ama uyarı veriyoruz
    model = None

# ============================================================
# API Endpointleri
# ============================================================

@app.get("/")
async def root():
    return {"message": "Traffic Sign Detection API is running!"}

@app.get("/health")
async def health_check():
    status = "ready" if model is not None else "model_not_found"
    return {"status": status}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Yüklenen görsel üzerinde trafik işareti tespiti yapar.
    
    Input: Image file
    Output: JSON detection results (label, confidence, bbox)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server")

    # 1. Dosya formatı kontrolü
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Dosya bir resim olmalıdır.")

    try:
        # 2. Resmi oku
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # 3. Model tahmini (Inference)
        # imgsz=640 eğitimdeki boyutla aynı olmalı
        results = model.predict(image, imgsz=640, conf=0.25)
        
        # 4. Sonuçları rules.md formatına dönüştür
        detections = []
        result = results[0]
        
        for box in result.boxes:
            # Koordinatları al [x1, y1, x2, y2]
            # rules.md 'bbox' bekliyor, genellikle [x, y, w, h] formatında verilir
            # Biz burada [x1, y1, x2, y2] döneceğiz (frontend'de çizim için daha kolay)
            # veya rules.md'deki standart [x, y, w, h] formatına çevirebiliriz.
            xyxy = box.xyxy[0].tolist()
            
            # Sınıf ID ve ismi
            cls_id = int(box.cls[0])
            label = result.names[cls_id]
            
            # Güven skoru
            confidence = float(box.conf[0])
            
            detections.append({
                "label": label,
                "confidence": round(confidence, 4),
                "bbox": [round(x, 2) for x in xyxy] # [left, top, right, bottom]
            })
            
        return detections

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tahmin hatası: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
