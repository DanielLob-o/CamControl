import cv2
import numpy as np
import sys
from ultralytics import YOLO
import os

# --- CONFIGURATION ---
SHOW_WINDOW = True 
PROCESS_EVERY_N_FRAMES = 10
TARGET_CLASSES = [0, 15] 
STREAM_URL = "http://192.168.1.161:8080/video" # mobile webcam stream

#0 = Person
#14 = Bird
#15 = Cat
#16 = Dog

# setup model, use yolov10n.pt if your computer is GPU optimized
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "yolov8n.pt")

print(f"🚀 Loading YOLO from: {model_path}", file=sys.stderr)

try:
    if os.path.exists(model_path):
        model = YOLO(model_path)
    else:
        print(f"⚠️ Model not found, downloading...", file=sys.stderr)
        model = YOLO("yolov8n.pt") # default model
except Exception as e:
    print(f"❌ Error loading model: {e}", file=sys.stderr)
    sys.exit(1)

def draw_hud(frame, center_x, center_y, width, height, err_x, err_y):
    """Draws a tactical crosshair and error data on the screen"""
    # Colors (RGB)
    GRAY = (100, 100, 100)
    CYAN = (255, 255, 0)
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    
    # center crosshair with gap
    gap = 5
    length = 30
    # Horizontal
    cv2.line(frame, (center_x - length, center_y), (center_x - gap, center_y), GRAY, 1)
    cv2.line(frame, (center_x + gap, center_y), (center_x + length, center_y), GRAY, 1)
    # Vertical
    cv2.line(frame, (center_x, center_y - length), (center_x, center_y - gap), GRAY, 1)
    cv2.line(frame, (center_x, center_y + gap), (center_x, center_y + length), GRAY, 1)
    
    # error bars for visal feedback
    if err_x is not None and err_y is not None:
        cv2.line(frame, (center_x, center_y), (center_x + err_x, center_y), RED, 2)
        cv2.line(frame, (center_x + err_x, center_y), (center_x + err_x, center_y + err_y), RED, 2)
        
        # Background box for text to make it readable
        cv2.rectangle(frame, (10, 10), (200, 70), (0, 0, 0), -1) # Filled black box
        cv2.putText(frame, f"ERROR X: {err_x}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, CYAN, 2)
        cv2.putText(frame, f"ERROR Y: {err_y}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, CYAN, 2)

def main():
    print(f"Python Worker Started (Stream Mode)", file=sys.stderr)
    print(f"Attempting to connect to: {STREAM_URL}", file=sys.stderr)

    frame_count = 0
    last_results = [] 

    # Setup Video Capture
    cap = cv2.VideoCapture(STREAM_URL)
    if not cap.isOpened():
        print("❌ Could not open video stream. Check URL and connection.", file=sys.stderr)
        return

    while True:
        try:
            # --- FRAME ACQUISITION ---
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame from stream", file=sys.stderr)
                break
            
            if frame is None: continue

            h, w, _ = frame.shape
            center_x, center_y = w // 2, h // 2

            frame_count += 1
            
            # --- AI PROCESSING ---
            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                results = model(frame, verbose=False)
                last_results = [] 

                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        if cls_id in TARGET_CLASSES:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                            err_x, err_y = cx - center_x, cy - center_y
                            
                            label_text = model.names[cls_id]
                            
                            last_results.append({
                                'box': [x1, y1, x2, y2],
                                'label': label_text,
                                'err_x': int(err_x),
                                'err_y': int(err_y),
                                'cx': cx, 
                                'cy': cy
                            })
                            # Track only first target
                            break 
            
            # --- VISUALIZATION LAYER ---
            active_err_x = None
            active_err_y = None

            if len(last_results) > 0:
                # Use the first target for the HUD error display
                active_err_x = last_results[0]['err_x']
                active_err_y = last_results[0]['err_y']

            if SHOW_WINDOW:
                draw_hud(frame, center_x, center_y, w, h, active_err_x, active_err_y)

            for item in last_results:
                if SHOW_WINDOW:
                    x1, y1, x2, y2 = item['box']
                    label = item['label']
                    # green box around target
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if SHOW_WINDOW:
                cv2.imshow("CamControl AI", frame)
                if cv2.waitKey(1) == ord('q'): break

        except Exception as e:
            print(f"Python Error: {e}", file=sys.stderr)
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()