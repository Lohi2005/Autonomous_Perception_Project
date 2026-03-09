from ultralytics import YOLO
import cv2

def load_object_detector():
    """Load YOLOv8 model"""
    return YOLO("models/yolov8n.pt")

def detect_objects(model, frame):
    """Detect objects and return annotated frame and detected objects"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb_frame, verbose=False)

    annotated = results[0].plot()
    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

    detected_objects = []
    for box, cls, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
        x1, y1, x2, y2 = box.int().tolist()
        detected_objects.append({
            "class_id": int(cls),
            "bbox": [x1, y1, x2, y2],
            "confidence": float(conf)
        })
    return annotated_bgr, detected_objects
