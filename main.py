import cv2
import os
import time
import numpy as np
from lane_detection import detect_lanes
from object_detection import load_object_detector, detect_objects
import tkinter as tk
from tkinter import filedialog

# ------------------------
# 1. Setup
# ------------------------
os.makedirs("output", exist_ok=True)
model = load_object_detector()
Kp = 0.1  # Steering proportional constant
stop_frames = 0  # Frames to wait at red light

# Load car icon (PNG with transparency)
car_icon_path = "assets/car_icon.png"
car_img = cv2.imread(car_icon_path, cv2.IMREAD_UNCHANGED)
if car_img is None:
    print(f"Car icon not found at {car_icon_path}")
    exit()
car_h, car_w = 100, 60  # Resize car icon
car_img = cv2.resize(car_img, (car_w, car_h))

# ------------------------
# 2. Video Upload
# ------------------------
root = tk.Tk()
root.withdraw()  # Hide GUI

video_path = filedialog.askopenfilename(
    title="Select Video File",
    filetypes=[("MP4 files","*.mp4"),("All files","*.*")]
)

if not video_path:
    print("No video selected. Exiting.")
    exit()

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

# ------------------------
# 3. Video Writer Setup (Slow Video)
# ------------------------
ret, frame = cap.read()
if not ret:
    print("Cannot read first frame of video.")
    exit()

height, width, _ = frame.shape
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output/perception_simulation_car.mp4", fourcc, 5, (width, height))

prev_time = 0

# ------------------------
# 4. Main Loop
# ------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_center = width // 2
    car_bottom = (frame_center, height - 50)

    # ---- Lane Detection ----
    lane_frame, lane_center = detect_lanes(frame)
    lane_offset = lane_center - frame_center
    steering_angle = Kp * lane_offset

    # ---- Object Detection ----
    result_frame, objects = detect_objects(model, lane_frame)

    throttle = 1.0
    brake = 0
    nearest_distance = 1.0  # normalized 0=close, 1=far
    red_light_detected = False

    # ---- Check obstacles and traffic signals ----
    for obj in objects:
        cls = obj["class_id"]
        x1, y1, x2, y2 = obj["bbox"]
        bbox_height = y2 - y1

        # Obstacles
        if cls in [0,1,2,3,5,7] and bbox_height > height*0.3:
            throttle = 0
            brake = 1
            nearest_distance = min(nearest_distance, height / bbox_height * 0.2)
            cv2.putText(result_frame, "OBSTACLE AHEAD!", (50,80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

        # Traffic signals / stop signs
        if cls in [9,11]:  # Example: stop sign / traffic light
            red_light_detected = True
            throttle = 0
            brake = 1
            stop_frames = 50  # wait for 50 frames (~10 sec at 5 FPS)
            cv2.putText(result_frame, "RED LIGHT!", (50,110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    # If waiting at red light
    if stop_frames > 0:
        throttle = 0
        brake = 1
        stop_frames -= 1

    # ---- Draw Car Icon (overlay PNG) ----
    car_x = int(frame_center + lane_offset - car_w//2)
    car_y = height - car_h - 30

    # Make sure car stays in frame
    car_x = max(0, min(car_x, width - car_w))

    # Overlay PNG with alpha
    for c in range(0,3):
        alpha = car_img[:,:,3]/255.0
        result_frame[car_y:car_y+car_h, car_x:car_x+car_w, c] = (
            alpha*car_img[:,:,c] + (1-alpha)*result_frame[car_y:car_y+car_h, car_x:car_x+car_w, c]
        )

    # ---- Draw Dynamic Lane Line (curve) ----
    lane_color = (0,255,0)
    cv2.line(result_frame, (int(lane_center), height), (int(lane_center), 0), lane_color, 2)

    # ---- Steering Arrow ----
    if abs(steering_angle) < 5: arrow_color = (0,255,0)
    elif abs(steering_angle) < 15: arrow_color = (0,255,255)
    else: arrow_color = (0,0,255)
    lane_target = (int(lane_center), height - 200)
    cv2.arrowedLine(result_frame, (frame_center, height - 50), lane_target, arrow_color, 5, tipLength=0.3)

    # ---- Obstacle Warning Bar ----
    bar_max_height = 200
    bar_height = int((1 - nearest_distance) * bar_max_height)
    cv2.rectangle(result_frame, (width-50, 50), (width-30, 50+bar_height), (0,0,255), -1)
    cv2.putText(result_frame, "Obstacle Distance", (width-200, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    # ---- Overlay Info ----
    cv2.putText(result_frame, f"Lane Offset: {int(lane_offset)} px", (50,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(result_frame, f"Steering Angle: {steering_angle:.2f}", (50,140),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    cv2.putText(result_frame, f"Throttle: {throttle}", (50,170),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    cv2.putText(result_frame, f"Brake: {brake}", (50,200),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # ---- FPS ----
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(result_frame, f"FPS: {int(fps)}", (width-150,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    # ---- Display & Save ----
    out.write(result_frame)
    if cv2.waitKey(200) & 0xFF == ord('q'):  # slow-motion display
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Demo complete. Output saved at output/perception_simulation_car.mp4")
