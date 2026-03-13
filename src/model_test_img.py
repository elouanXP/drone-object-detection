from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd

from config import OUTPUTS_DIR, DATA_YAML, MODEL_PATH

CLASS_NAMES = ["pedestrian", "vehicle", "bike"]

def load_model(model_path=MODEL_PATH):
    model = YOLO(model_path)
    return model

def visualize_predictions(model, num_images=6):
    
    val_img_dir = Path('data/processed/val/images')
    img_files = list(val_img_dir.glob('*.jpg'))
    np.random.seed(42)
    np.random.shuffle(img_files)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes[:num_images]):
        img_path = img_files[i]

        results = model.predict(img_path, conf=0.25, verbose=False)

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            
            for box, cls, conf in zip(boxes, classes, confs):
                x1, y1, x2, y2 = map(int, box)
                label = f"{CLASS_NAMES[int(cls)]} {conf:.2f}"

                color = tuple(np.random.randint(0, 255, 3).tolist())
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        ax.imshow(img)
        ax.set_title(f"{img_path.name}\n{len(results[0].boxes)} detections")
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUTS_DIR}/plots/predictions_validation_yolo26.png', 
                dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

def evaluate_model(model):
    
    metrics = model.val(data=DATA_YAML, split='val', verbose=True)

    results = {
        'mAP50': metrics.box.map50,
        'mAP50-95': metrics.box.map,
        'Precision': metrics.box.mp,
        'Recall': metrics.box.mr
    }
    
    print("\n=== Global metrics ===")
    for metric, value in results.items():
        print(f"{metric:12s}: {value:.4f}")
 
    print("\n=== Performance for each class ===")
    class_metrics = []
    
    for i, class_name in enumerate(CLASS_NAMES):
        if i < len(metrics.box.ap50):
            class_metrics.append({
                'Class': class_name,
                'mAP50': metrics.box.ap50[i],
                'Precision': metrics.box.p[i] if i < len(metrics.box.p) else 0,
                'Recall': metrics.box.r[i] if i < len(metrics.box.r) else 0
            })
    
    df = pd.DataFrame(class_metrics)
    df = df.sort_values('mAP50', ascending=False)
    print(df.to_string(index=False))
    
    return metrics

if __name__ == "__main__":
    model = load_model()
    metrics = evaluate_model(model)
    visualize_predictions(model, num_images=6)