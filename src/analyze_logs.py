import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

def parse_log_file(log_path='outputs/logs/api.log'):
    """Parse le fichier de logs API"""
    
    records = []
    
    with open(log_path, 'r') as f:
        for line in f:
            if ' - INFO - {' in line:
                # Extraire le JSON
                json_start = line.find('{')
                json_str = line[json_start:]
                
                try:
                    data = json.loads(json_str)
                    records.append(data)
                except json.JSONDecodeError:
                    continue
    
    return pd.DataFrame(records)

def analyze_api_performance(df):
    """Analyse les performances de l'API"""
    
    print("=== API Performance Summary ===\n")
    
    print(f"Total requests: {len(df)}")
    print(f"Avg inference time: {df['inference_time_ms'].mean():.2f} ms")
    print(f"Median inference time: {df['inference_time_ms'].median():.2f} ms")
    print(f"Avg FPS: {df['fps'].mean():.1f}")
    print(f"Avg detections per image: {df['num_detections'].mean():.1f}")
    
    print("\n=== Detection Statistics ===\n")
    
    # Flatten class distribution
    all_classes = {}
    for class_dist in df['class_distribution']:
        for cls, count in class_dist.items():
            all_classes[cls] = all_classes.get(cls, 0) + count
    
    print("Total detections by class:")
    for cls, count in sorted(all_classes.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cls}: {count}")
    
    # Visualisations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Inference time distribution
    ax = axes[0, 0]
    ax.hist(df['inference_time_ms'], bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(df['inference_time_ms'].median(), color='red', linestyle='--', 
               label=f'Median: {df["inference_time_ms"].median():.1f} ms')
    ax.set_xlabel('Inference Time (ms)')
    ax.set_ylabel('Frequency')
    ax.set_title('API Inference Time Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. FPS distribution
    ax = axes[0, 1]
    ax.hist(df['fps'], bins=30, edgecolor='black', alpha=0.7, color='green')
    ax.axvline(df['fps'].median(), color='red', linestyle='--',
               label=f'Median: {df["fps"].median():.1f} FPS')
    ax.set_xlabel('FPS')
    ax.set_ylabel('Frequency')
    ax.set_title('API FPS Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. Detections per image
    ax = axes[1, 0]
    ax.hist(df['num_detections'], bins=30, edgecolor='black', alpha=0.7, color='orange')
    ax.set_xlabel('Detections per Image')
    ax.set_ylabel('Frequency')
    ax.set_title('Detections Distribution')
    ax.grid(alpha=0.3)
    
    # 4. Class distribution
    ax = axes[1, 1]
    classes = list(all_classes.keys())
    counts = list(all_classes.values())
    ax.barh(classes, counts)
    ax.set_xlabel('Total Detections')
    ax.set_title('Detection Distribution by Class')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/plots/api_performance.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Visualisations sauvegardées dans outputs/plots/api_performance.png")

if __name__ == "__main__":
    log_path = Path('outputs/logs/api.log')
    
    if not log_path.exists():
        print(f"Fichier de logs introuvable: {log_path}")
        print("Lance l'API et teste quelques images d'abord.")
    else:
        df = parse_log_file(log_path)
        
        if len(df) == 0:
            print("Aucune requête API enregistrée dans les logs")
        else:
            analyze_api_performance(df)