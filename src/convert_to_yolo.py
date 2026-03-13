import json
from pathlib import Path
from tqdm import tqdm

CLASS_MAPPING = {
    "pedestrian": 0,
    "people": 0,
    "bicycle": 2,
    "car": 1,
    "van": 1,
    "truck": 1,
    "tricycle": 2,
    "awning-tricycle": 2,
    "bus": 1,
    "motor": 1,
    "ignored region": -1
}

def convert_supervisely_to_yolo(json_path, img_width, img_height):
    """
    Convert ann Supervisely JSON --> YOLO format
    
    Returns:
        list: List of strings in YOLO format
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    yolo_lines = []
    
    for obj in data['objects']:
        class_name = obj['classTitle']
        
        if class_name == "ignored region":
            continue
            
        if class_name not in CLASS_MAPPING:
            continue
            
        class_id = CLASS_MAPPING[class_name]

        points = obj['points']['exterior']
        x1, y1 = points[0]
        x2, y2 = points[1]

        x_center = ((x1 + x2) / 2) / img_width
        y_center = ((y1 + y2) / 2) / img_height
        width = abs(x2 - x1) / img_width
        height = abs(y2 - y1) / img_height
        
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return yolo_lines

def process_dataset(split='train'):
    """
    Convert a split (train/val) to YOLO format
    """
    base_dir = Path(f'data/raw/{split}')
    img_dir = base_dir / 'img'
    ann_dir = base_dir / 'ann'

    output_dir = Path(f'data/processed/{split}')
    output_img_dir = output_dir / 'images'
    output_label_dir = output_dir / 'labels'
    
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    ann_files = list(ann_dir.glob('*.json'))
    
    print(f"Conversion de {len(ann_files)} images ({split})...")
    
    for ann_file in tqdm(ann_files):

        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        img_height = data['size']['height']
        img_width = data['size']['width']

        yolo_lines = convert_supervisely_to_yolo(ann_file, img_width, img_height)
        
        json_stem = ann_file.stem
        img_name = Path(json_stem).stem

        label_file = output_label_dir / f'{img_name}.txt'

        with open(label_file, 'w') as f:
            f.write('\n'.join(yolo_lines))

        img_file = img_dir / json_stem
        if img_file.exists():
            import shutil
            shutil.copy(img_file, output_img_dir / f'{img_name}.jpg')
    
    print(f"{split} converti → {output_dir}")

if __name__ == "__main__":
    process_dataset('train')
    process_dataset('val')