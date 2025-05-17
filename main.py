import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyttsx3
import time
from collections import Counter
import numpy as np

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Initialize MediaPipe Object Detection
model_path = 'efficientdet_lite0.tflite'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.ObjectDetectorOptions(
    base_options=base_options,
    score_threshold=0.5,
    category_allowlist=[
            "person", "car", "truck", "motorcycle", "bicycle",
            "tree", "bench", "traffic light", "stop sign"
    ]
)
object_detector = vision.ObjectDetector.create_from_options(options)

# Camera parameters (needed for distance estimation)
KNOWN_WIDTHS = {
    "car": 1.8,  # Average car width in meters
    "truck": 2.5,  # Average truck width
    "motorcycle": 0.8,
    "bicycle": 0.7,
    "person": 0.5,
    "tree": 1.0  # Average tree trunk width
}
FOCAL_LENGTH = 1000  # You should calibrate this for your camera

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
last_announce_time = time.time()


def estimate_distance(object_width_pixels, object_type):
    """Estimate distance to object using pixel width"""
    if object_type not in KNOWN_WIDTHS:
        return None
    real_width = KNOWN_WIDTHS[object_type]
    distance = (real_width * FOCAL_LENGTH) / object_width_pixels
    return round(distance, 1)


def classify_vehicle(label):
    """Classify vehicles based on labels"""
    if label == "car":
        return "4-wheeler"
    elif label == "truck":
        return "6-wheeler"
    elif label in ["motorcycle", "bicycle"]:
        return "2-wheeler"
    return label


def draw_object_detections(frame, detections):
    annotated_image = frame.copy()
    height, width = frame.shape[:2]

    for detection in detections.detections:
        bbox = detection.bounding_box
        category = detection.categories[0]
        label = classify_vehicle(category.category_name)
        score = round(category.score, 2)

        # Calculate distance if we know the object type
        distance = estimate_distance(bbox.width, category.category_name)

        # Draw bounding box
        color = (0, 255, 0) if distance and distance > 5 else (0, 0, 255)
        cv2.rectangle(annotated_image,
                      (bbox.origin_x, bbox.origin_y),
                      (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height),
                      color, 2)

        # Draw label with distance
        label_text = f"{label} ({score})"
        if distance:
            label_text += f" {distance}m"
        cv2.putText(annotated_image, label_text,
                    (bbox.origin_x, bbox.origin_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return annotated_image


def announce_detections(detections):
    global last_announce_time

    if time.time() - last_announce_time < 5:
        return

    if detections:
        objects = []
        for detection in detections.detections:
            category = detection.categories[0]
            if category.score > 0.5:
                obj_type = classify_vehicle(category.category_name)
                distance = estimate_distance(detection.bounding_box.width, category.category_name)
                if distance:
                    objects.append(f"{obj_type} {distance}m")
                else:
                    objects.append(obj_type)

        if objects:
            announcement = "Ahead: " + ", ".join(objects[:3])
            engine.say(announcement)
            engine.runAndWait()
            last_announce_time = time.time()


def detect_speed_breakers(frame):
    """Simple speed breaker detection using edge detection"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                            minLineLength=100, maxLineGap=10)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y1 - y2) < 10 and abs(x1 - x2) > 100:  # Horizontal line
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                return True
    return False


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect speed breakers
    speed_breaker_detected = detect_speed_breakers(frame)
    if speed_breaker_detected:
        cv2.putText(frame, "Speed Breaker!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Object detection
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = object_detector.detect(mp_image)

    # Draw detections
    frame_with_detections = draw_object_detections(frame, detection_result)

    # Audio feedback
    announce_detections(detection_result)
    if speed_breaker_detected and time.time() - last_announce_time > 1:
        engine.say("Speed breaker ahead")
        engine.runAndWait()
        last_announce_time = time.time()

    cv2.imshow('Advanced Obstacle Detection', frame_with_detections)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()