from ultralytics import YOLO
import cv2
from pathlib import Path
import time

from config import OUTPUTS_DIR, MODEL_PATH

def inference_on_video(video_path, model_path=MODEL_PATH, 
                       conf_threshold=0.25, save_output=True):

    model = YOLO(model_path)
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Error {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo : {video_path.name}")
    print(f"Resolution : {width}x{height}")
    print(f"FPS : {fps}")
    print(f"Total frames : {total_frames}")

    if save_output:
        output_path = Path(OUTPUTS_DIR) / 'videos' / f"detected_yolo26{video_path.name}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_count = 0
    total_time = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break

        start_time = time.time()
        results = model.predict(frame, conf=conf_threshold, verbose=False)
        inference_time = time.time() - start_time
        total_time += inference_time

        annotated_frame = results[0].plot()

        num_detections = len(results[0].boxes)
        current_fps = 1 / inference_time if inference_time > 0 else 0
        
        cv2.putText(annotated_frame, f"FPS: {current_fps:.1f} | Detections: {num_detections}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if save_output:
            out.write(annotated_frame)
        
        frame_count += 1

        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            avg_fps = frame_count / total_time
            print(f"Progression : {progress:.1f}% | Avg FPS : {avg_fps:.1f}")

    cap.release()
    if save_output:
        out.release()

    avg_fps = frame_count / total_time
    print(f"Frames : {frame_count}")
    print(f"Avg FPS : {avg_fps:.1f}")
    
    if save_output:
        print(f"Saved: {output_path}")
    
    return output_path if save_output else None

if __name__ == "__main__":
    video_path = Path("data/test_video.mp4")
    
    if video_path.exists():
        inference_on_video(video_path)
    else:
        print(f"No video {video_path}")