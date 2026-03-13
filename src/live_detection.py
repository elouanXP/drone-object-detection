import cv2
import subprocess
import time
import threading
import collections
from ultralytics import YOLO

FFMPEG        = r"C:/ffmpeg/ffmpeg-8.0.1-essentials_build/bin/ffmpeg.exe"
MODEL_PATH    = r"../models/yolo26n_visdrone/weights/best.pt"
DRONE_TCP     = "tcp://192.168.30.1:8080"
UDP_LOCAL     = "udp://127.0.0.1:5000?fifo_size=1000000&overrun_nonfatal=1"
YOLO_EVERY_N  = 3
CONF          = 0.3
WINDOW        = "Drone YOLO - VisDrone"

model = YOLO(MODEL_PATH)

class SystemMetrics:
    def __init__(self, window=60):
        self.frame_times = collections.deque(maxlen=window)
        self.infer_times = collections.deque(maxlen=window)
        self.dropped     = 0
        self.total       = 0
        self._last       = None

    def on_frame(self):
        now = time.perf_counter()
        if self._last:
            self.frame_times.append(now - self._last)
        self._last = now
        self.total += 1

    def on_infer(self, s):
        self.infer_times.append(s * 1000)

    def on_drop(self):
        self.dropped += 1

    @property
    def display_fps(self):
        if len(self.frame_times) < 2:
            return 0.0
        return 1.0 / (sum(self.frame_times) / len(self.frame_times))

    @property
    def mean_infer_ms(self):
        return sum(self.infer_times) / len(self.infer_times) if self.infer_times else 0.0

    @property
    def drop_rate(self):
        return self.dropped / self.total * 100 if self.total else 0.0

metrics = SystemMetrics()

proc = subprocess.Popen([
    FFMPEG,
    "-fflags", "nobuffer+discardcorrupt",
    "-flags", "low_delay",
    "-i", DRONE_TCP,
    "-f", "mpegts", "-vcodec", "copy",
    UDP_LOCAL
], stderr=subprocess.DEVNULL)

print("Loading stream...")
time.sleep(3)

cap = cv2.VideoCapture(UDP_LOCAL, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

latest_frame = [None]
lock         = threading.Lock()
running      = [True]

def drain_buffer():
    while running[0]:
        try:
            ret, frame = cap.read()
            if ret:
                with lock:
                    latest_frame[0] = frame
            else:
                metrics.on_drop()
        except cv2.error:
            break

threading.Thread(target=drain_buffer, daemon=True).start()
print("Flux ouvert ! Q pour quitter")

def draw_overlay(frame, boxes, ids):
    display = frame.copy()
    if boxes is not None:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls   = int(box.cls[0])
            conf  = float(box.conf[0])
            label = model.names[cls]
            tid   = int(ids[i]) if ids is not None and i < len(ids) else -1
            color = (0, 255, 0) if tid == -1 else (
                int((tid * 67)  % 255),
                int((tid * 113) % 255),
                int((tid * 181) % 255)
            )
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            text = f"{label} {conf:.2f}" + (f" #{tid}" if tid != -1 else "")
            cv2.putText(display, text, (x1, max(y1 - 6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    for i, line in enumerate([
        f"FPS    : {metrics.display_fps:5.1f}",
        f"Infer  : {metrics.mean_infer_ms:5.1f} ms",
        f"Drop   : {metrics.drop_rate:5.1f} %",
        f"Frames : {metrics.total}",
    ]):
        cv2.putText(display, line, (10, 20 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1)
    return display

frame_count = 0
last_boxes  = None
last_ids    = None
out         = None

try:
    while True:
        with lock:
            frame = latest_frame[0]
        if frame is None:
            time.sleep(0.01)
            continue

        metrics.on_frame()
        frame_count += 1

        if out is None:
            h, w = frame.shape[:2]
            out = cv2.VideoWriter(
                f"session_{int(time.time())}.mp4",
                cv2.VideoWriter_fourcc(*'mp4v'),
                25.0, (w, h)
            )
            print(f"Enregistrement démarré ({w}x{h})")

        if frame_count % YOLO_EVERY_N == 0:
            t0 = time.perf_counter()
            results = model.track(
                frame, persist=True,
                tracker="bytetrack.yaml",
                verbose=False, conf=CONF
            )
            metrics.on_infer(time.perf_counter() - t0)
            r          = results[0]
            last_boxes = r.boxes if len(r.boxes) > 0 else None
            last_ids   = (r.boxes.id if last_boxes is not None
                          and r.boxes.id is not None else None)

        display = draw_overlay(frame, last_boxes, last_ids)
        cv2.imshow(WINDOW, display)
        out.write(display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

finally:
    print("\n── Résumé session ───────────────────────────────")
    print(f"  Frames totales          : {metrics.total}")
    print(f"  FPS moyen               : {metrics.display_fps:.1f}")
    print(f"  Latence inférence (moy) : {metrics.mean_infer_ms:.1f} ms")
    print(f"  Frames droppées         : {metrics.dropped} ({metrics.drop_rate:.1f} %)")
    print("─────────────────────────────────────────────────")
    running[0] = False
    if out is not None:
        out.release()
    proc.terminate()
    cap.release()
    cv2.destroyAllWindows()