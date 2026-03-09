import cv2
import numpy as np

def detect_lanes(frame):
    """Detect lanes and return annotated frame"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (0, height),
        (width, height),
        (width//2, int(height*0.6)),
        (width//2, int(height*0.6))
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 40, minLineLength=40, maxLineGap=100)
    line_image = np.zeros_like(frame)

    left_x = []
    right_x = []

    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(line_image, (x1,y1),(x2,y2),(0,255,0),5)
            # Separate left/right lanes by position
            if x1 < width//2 and x2 < width//2:
                left_x.append((x1+x2)//2)
            elif x1 > width//2 and x2 > width//2:
                right_x.append((x1+x2)//2)

    combined = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    # Compute lane center
    if left_x and right_x:
        lane_center = (np.mean(left_x) + np.mean(right_x)) / 2
    else:
        lane_center = width / 2  # fallback

    return combined, lane_center
