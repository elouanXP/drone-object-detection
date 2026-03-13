from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO
import uvicorn
import logging
from datetime import datetime
from pathlib import Path
import time
import json

log_dir = Path("outputs/logs")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'api.log'),
        logging.StreamHandler()
    ]
)

model = YOLO("models/yolo26n_visdrone/weights/best.pt")

app = FastAPI(
    title="Drone Object Detection API",
    description="Drone Object Detection API",
    version="1.0.0"
)

CLASS_NAMES = ["pedestrian", "vehicle", "bike"]

@app.get("/")
def root():
    return {
        "message": "Drone Object Detection API",
        "model": "YOLOv8n",
        "dataset": "VisDrone",
        "endpoints": {
            "/predict": "POST - Upload image for detection",
            "/predict/annotated": "POST - Get annotated image",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    conf_threshold: float = 0.25
):
    request_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        logging.warning(f"[{request_id}] Invalid file format: {file.content_type}")
        raise HTTPException(400, "Format non supporté. Utilisez jpg ou png")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        logging.error(f"[{request_id}] Failed to decode image")
        raise HTTPException(400, "Failed to decode image")

    start_time = time.time()
    results = model.predict(img, conf=conf_threshold, verbose=False)
    inference_time = time.time() - start_time

    detections = []
    class_counts = {}
    
    if len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        
        for box, cls, conf in zip(boxes, classes, confs):
            x1, y1, x2, y2 = box
            class_name = CLASS_NAMES[int(cls)]
            
            detections.append({
                "class": class_name,
                "confidence": float(conf),
                "bbox": {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2)
                }
            })

            class_counts[class_name] = class_counts.get(class_name, 0) + 1

    log_data = {
        "request_id": request_id,
        "filename": file.filename,
        "image_size": {"width": img.shape[1], "height": img.shape[0]},
        "num_detections": len(detections),
        "class_distribution": class_counts,
        "conf_threshold": conf_threshold,
        "inference_time_ms": round(inference_time * 1000, 2),
        "fps": round(1 / inference_time, 2)
    }
    
    logging.info(json.dumps(log_data))
    
    return JSONResponse({
        "request_id": request_id,
        "filename": file.filename,
        "image_size": {
            "width": img.shape[1],
            "height": img.shape[0]
        },
        "num_detections": len(detections),
        "class_distribution": class_counts,
        "inference_time_ms": round(inference_time * 1000, 2),
        "detections": detections
    })

@app.post("/predict/annotated")
async def predict_annotated(
    file: UploadFile = File(...),
    conf_threshold: float = 0.25
):

    request_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        logging.warning(f"[{request_id}] Invalid file format: {file.content_type}")
        raise HTTPException(400, "Format non supporté")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        logging.error(f"[{request_id}] Failed to decode image")
        raise HTTPException(400, "Failed to decode image")

    start_time = time.time()
    results = model.predict(img, conf=conf_threshold, verbose=False)
    inference_time = time.time() - start_time

    num_detections = len(results[0].boxes)
    log_data = {
        "request_id": request_id,
        "endpoint": "/predict/annotated",
        "filename": file.filename,
        "num_detections": num_detections,
        "inference_time_ms": round(inference_time * 1000, 2)
    }
    logging.info(json.dumps(log_data))

    annotated_img = results[0].plot()

    _, buffer = cv2.imencode('.jpg', annotated_img)
    io_buf = BytesIO(buffer)
    
    return StreamingResponse(io_buf, media_type="image/jpeg")

if __name__ == "__main__":
    print("http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)