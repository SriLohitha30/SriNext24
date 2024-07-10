import cv2
import numpy as np
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
def draw_lines(img, lines):
    if lines is None:
        return img
    img = np.copy(img)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 10)
    
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    return img

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    height, width = frame.shape[:2]
    vertices = np.array([[(0, height), (width / 2, height / 2), (width, height)]], dtype=np.int32)
    roi_edges = region_of_interest(edges, vertices)
    
    lines = cv2.HoughLinesP(roi_edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=150)
    frame_with_lines = draw_lines(frame, lines)
    
    return frame_with_lines

# Read video
cap = cv2.VideoCapture('path_to_your_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_with_lines = process_frame(frame)
    cv2.imshow('Lane Line Detection', frame_with_lines)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
