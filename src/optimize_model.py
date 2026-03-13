from ultralytics import YOLO
from pathlib import Path
import time
import cv2
from config import MODEL_PATH

def export_to_onnx(model_path=MODEL_PATH):

    model = YOLO(model_path)
    onnx_path = model.export(format="onnx", simplify=True)
    return onnx_path

def benchmark_models(pt_model_path, onnx_model_path, test_image_path):
    print("\n=== PyTorch vs ONNX ===\n")
    img = cv2.imread(str(test_image_path))
    model_pt = YOLO(pt_model_path)
    
    times_pt = []
    for i in range(10):
        start = time.time()
        results = model_pt.predict(img, verbose=False)
        times_pt.append(time.time() - start)
    
    avg_time_pt = sum(times_pt) / len(times_pt)
    fps_pt = 1 / avg_time_pt
    
    print(f"  Avg Time: {avg_time_pt*1000:.2f} ms")
    print(f"  FPS : {fps_pt:.1f}")

    model_onnx = YOLO(onnx_model_path)
    
    times_onnx = []
    for i in range(10):
        start = time.time()
        results = model_onnx.predict(img, verbose=False)
        times_onnx.append(time.time() - start)
    
    avg_time_onnx = sum(times_onnx) / len(times_onnx)
    fps_onnx = 1 / avg_time_onnx
    
    print(f"  Avg Time : {avg_time_onnx*1000:.2f} ms")
    print(f"  FPS : {fps_onnx:.1f}")

    speedup = avg_time_pt / avg_time_onnx
    print(f"Speedup ONNX : {speedup:.2f}x faster")
    
    return {
        'pt': {'time': avg_time_pt, 'fps': fps_pt},
        'onnx': {'time': avg_time_onnx, 'fps': fps_onnx},
        'speedup': speedup
    }

if __name__ == "__main__":
    pt_path = MODEL_PATH
    onnx_path = export_to_onnx(pt_path)
    test_img = "data/processed/val/images/0000001_02999_d_0000005.jpg"
    
    if Path(test_img).exists():
        benchmark_models(pt_path, onnx_path, test_img)
    else:
        print(f"missing img")