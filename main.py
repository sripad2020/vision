import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyttsx3
import speech_recognition as sr
import pytesseract
import time
import numpy as np
from collections import deque
import geocoder
import math
import socket
import threading
import json
import uuid
import logging
from datetime import datetime

# Initialize logging
logging.basicConfig(
    filename=f"caregiver_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Initialize MediaPipe Object Detection for general objects
model_path = 'efficientdet_lite0.tflite'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.ObjectDetectorOptions(
    base_options=base_options,
    score_threshold=0.5,
    category_allowlist=[
        "person", "car", "truck", "motorcycle", "bicycle",
        "tree", "bench", "traffic light", "stop sign",
        "pole", "debris", "barrier"
    ]
)
object_detector = vision.ObjectDetector.create_from_options(options)

# Initialize MediaPipe Object Detection for traffic signs
traffic_sign_model_path = 'traffic_sign_model.tflite'  # Replace with actual traffic sign model
traffic_sign_options = vision.ObjectDetectorOptions(
    base_options=python.BaseOptions(model_asset_path=traffic_sign_model_path),
    score_threshold=0.6,
    category_allowlist=[
        "stop", "yield", "speed limit 30", "speed limit 50", "no entry",
        "no parking", "no u-turn", "one way", "pedestrian crossing",
        "road work", "traffic signal ahead", "school zone",
        "bike lane", "left turn only", "right turn only"
    ]
)
traffic_sign_detector = vision.ObjectDetector.create_from_options(traffic_sign_options)

# Initialize MediaPipe Pose for behavior analysis
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Camera parameters
KNOWN_WIDTHS = {
    "car": 1.8, "truck": 2.5, "motorcycle": 0.8, "bicycle": 0.7,
    "person": 0.5, "tree": 1.0, "stop sign": 0.75, "bench": 1.5,
    "traffic light": 0.5, "pole": 0.3, "debris": 0.5, "barrier": 1.0,
    "pothole": 0.5, "curb": 0.3, "stairs": 1.0
}
FOCAL_LENGTH = 1000  # Calibrate for your camera
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# User and caregiver data
user_data = {
    "user_id": str(uuid.uuid4()),
    "caregiver_phone": "caregiver_phone_number",
    "destination": None,
    "location": None
}

# Behavior tracking
user_positions = deque(maxlen=10)
user_head_orientations = deque(maxlen=5)
user_speeds = deque(maxlen=10)
prev_person_positions = deque(maxlen=10)
last_announce_time = time.time()
emergency_detected = False

# UDP socket for video streaming
CAREGIVER_IP = "127.0.0.1"  # Replace with caregiver's IP
CAREGIVER_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def speak(text):
    """Speak text"""
    engine.say(text)
    engine.runAndWait()
    logging.info(f"Spoken: {text}")

def listen_for_command():
    """Listen for voice input"""
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        speak("Listening...")
        try:
            audio = recognizer.listen(source, timeout=5)
            command = recognizer.recognize_google(audio).lower()
            logging.info(f"Voice command: {command}")
            return command
        except (sr.WaitTimeoutError, sr.UnknownValueError, sr.RequestError):
            logging.info("No voice command recognized")
            return None

def estimate_distance(object_width_pixels, object_type):
    """Estimate distance to object"""
    if object_type not in KNOWN_WIDTHS:
        return None
    real_width = KNOWN_WIDTHS[object_type]
    distance = (real_width * FOCAL_LENGTH) / object_width_pixels
    return round(distance, 1)

def estimate_depth(landmarks, frame_width):
    """Estimate depth using shoulder width"""
    if not landmarks:
        return None
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    shoulder_width_pixels = abs(left_shoulder.x - right_shoulder.x) * frame_width
    if shoulder_width_pixels == 0:
        return None
    real_shoulder_width = 0.4
    depth = (real_shoulder_width * FOCAL_LENGTH) / shoulder_width_pixels
    return round(depth, 1)

def classify_vehicle(label):
    """Classify vehicles"""
    if label == "car":
        return "4-wheeler"
    elif label == "truck":
        return "6-wheeler"
    elif label in ["motorcycle", "bicycle"]:
        return "2-wheeler"
    return label

def analyze_user_behavior(pose_landmarks, frame_width, frame_height, prev_positions, prev_orientations, speeds):
    """Analyze 10 user behaviors"""
    if not pose_landmarks:
        return "No user detected", 0, None

    left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    nose = pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
    left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_knee = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
    right_knee = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]

    # 1-3: Head orientation
    shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2 * frame_width
    nose_x = nose.x * frame_width
    head_orientation = "Looking forward"
    if nose_x < shoulder_mid_x - 50:
        head_orientation = "Looking left"
    elif nose_x > shoulder_mid_x + 50:
        head_orientation = "Looking right"
    prev_orientations.append(head_orientation)
    smoothed_orientation = max(set(prev_orientations), key=list(prev_orientations).count)

    # 4-7: Body movement
    hip_mid_x = (left_hip.x + right_hip.x) / 2 * frame_width
    hip_mid_y = (left_hip.y + right_hip.y) / 2 * frame_height
    movement = "Stationary"
    speed = 0
    if prev_positions:
        prev_x, prev_y = prev_positions[-1]
        velocity_x = hip_mid_x - prev_x
        velocity_y = hip_mid_y - prev_y
        speed = np.sqrt(velocity_x**2 + velocity_y**2) / frame_width * 100
        speeds.append(speed)
        avg_speed = np.mean(speeds)

        if avg_speed > 5:
            if abs(velocity_x) > abs(velocity_y):
                movement = "Turning left" if velocity_x < -10 else "Turning right"
            else:
                movement = "Moving forward" if velocity_y < -10 else "Moving backward"
        elif avg_speed > 2:
            movement = "Slowing down"
        else:
            movement = "Stationary"

    prev_positions.append((hip_mid_x, hip_mid_y))

    # 8-9: Gestures
    gesture = None
    shoulder_height = (left_shoulder.y + right_shoulder.y) / 2 * frame_height
    if left_wrist.y * frame_height < shoulder_height - 50 and left_wrist.visibility > 0.7:
        gesture = "Waving left hand"
    elif right_wrist.y * frame_height < shoulder_height - 50 and right_wrist.visibility > 0.7:
        gesture = "Waving right hand"
    elif abs(left_wrist.x - right_wrist.x) * frame_width > 100 and left_wrist.y * frame_height < shoulder_height:
        gesture = "Pointing"

    # 10: Crouching
    hip_height = (left_hip.y + right_hip.y) / 2 * frame_height
    knee_height = (left_knee.y + right_knee.y) / 2 * frame_height
    crouching = "Crouching" if knee_height < hip_height + 50 and avg_speed < 2 else None

    behavior = smoothed_orientation
    if movement != "Stationary":
        behavior += f" and {movement}"
    if gesture:
        behavior += f" and {gesture}"
    if crouching:
        behavior += f" and {crouching}"

    logging.info(f"User behavior: {behavior}, Speed: {avg_speed:.2f}")
    return behavior, avg_speed, gesture

def predict_person_behavior(pose_landmarks, prev_positions, frame_width):
    """Predict person behavior"""
    if not pose_landmarks:
        return "No person detected", None
    left_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    hip_mid_x = (left_hip.x + right_hip.x) / 2 * frame_width
    if prev_positions:
        prev_x = prev_positions[-1]
        velocity = hip_mid_x - prev_x
        if velocity > 10:
            return "Moving right", hip_mid_x
        elif velocity < -10:
            return "Moving left", hip_mid_x
        return "Stationary", hip_mid_x
    return "Stationary", hip_mid_x

def detect_speed_breakers(frame):
    """Detect speed breakers"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                            minLineLength=100, maxLineGap=10)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y1 - y2) < 10 and abs(x1 - x2) > 100:
                return True
    return False

def detect_potholes(frame):
    """Detect potholes"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if 100 < area < 1000:
            return True
    return False

def detect_curbs(frame):
    """Detect curbs"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                            minLineLength=50, maxLineGap=5)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y1 - y2) > 50 and abs(x1 - x2) < 20:
                return True
    return False

def detect_stairs(frame):
    """Detect stairs"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                            minLineLength=80, maxLineGap=10)
    if lines is not None:
        horizontal_lines = sum(1 for line in lines if abs(line[0][1] - line[0][3]) < 10)
        if horizontal_lines > 3:
            return True
    return False

def is_road_context(frame, traffic_sign_detected, speed_breaker):
    """Detect road context"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    line_count = len(cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                                     minLineLength=100, maxLineGap=10) or [])
    return line_count > 5 or traffic_sign_detected or speed_breaker

def read_text(frame):
    """Perform OCR"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config='--psm 6')
    text = text.strip() if text.strip() else None
    if text:
        logging.info(f"OCR text: {text}")
    return text

def get_directions(destination):
    """Mock navigation directions"""
    g = geocoder.osm(destination)
    if g.ok:
        user_loc = geocoder.ip('me').latlng
        dest_loc = g.latlng
        distance = math.sqrt((dest_loc[0] - user_loc[0])**2 + (dest_loc[1] - user_loc[1])**2) * 111
        return {"distance": round(distance, 1), "steps": ["Turn left in 50 meters", "Continue straight for 200 meters"]}
    return None

def call_caregiver():
    """Mock caregiver call"""
    speak("Calling caregiver (mock implementation)")
    logging.info("Caregiver call initiated (mock)")
    return True

def draw_object_detections(frame, detections, traffic_sign_detections, user_behavior, navigation_steps, road_context):
    """Draw AR overlays in road context"""
    annotated_image = frame.copy()
    height, width = frame.shape[:2]

    if road_context:
        # General objects
        for detection in detections.detections:
            bbox = detection.bounding_box
            category = detection.categories[0]
            label = classify_vehicle(category.category_name)
            score = round(category.score, 2)
            distance = estimate_distance(bbox.width, category.category_name)

            color = (0, 255, 0) if distance and distance > 5 else (0, 0, 255)
            cv2.rectangle(annotated_image, (bbox.origin_x, bbox.origin_y),
                          (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height), color, 2)
            label_text = f"{label} ({score})"
            if distance:
                label_text += f" {distance}m"
            cv2.putText(annotated_image, label_text, (bbox.origin_x, bbox.origin_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Traffic signs
        for detection in traffic_sign_detections.detections:
            bbox = detection.bounding_box
            category = detection.categories[0]
            label = category.category_name
            score = round(category.score, 2)
            distance = estimate_distance(bbox.width, "stop sign")

            color = (255, 255, 0)
            cv2.rectangle(annotated_image, (bbox.origin_x, bbox.origin_y),
                          (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height), color, 2)
            label_text = f"{label} ({score})"
            if distance:
                label_text += f" {distance}m"
            cv2.putText(annotated_image, label_text, (bbox.origin_x, bbox.origin_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Navigation arrows
        if navigation_steps:
            cv2.arrowedLine(annotated_image, (50, 50), (100, 50), (0, 255, 255), 3)
            cv2.putText(annotated_image, navigation_steps[0], (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # User behavior
    cv2.putText(annotated_image, f"User: {user_behavior}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return annotated_image

def announce_detections(detections, traffic_sign_detections, user_behavior, person_behavior, text, obstacles, navigation_steps):
    """Announce detections and alerts"""
    global last_announce_time
    if time.time() - last_announce_time < 5:
        return

    announcements = []
    for detection in detections.detections:
        category = detection.categories[0]
        if category.score > 0.5:
            obj_type = classify_vehicle(category.category_name)
            distance = estimate_distance(detection.bounding_box.width, category.category_name)
            if distance:
                announcements.append(f"{obj_type} {distance} meters")
                logging.info(f"Detected: {obj_type}, Distance: {distance}m")
            else:
                announcements.append(obj_type)
                logging.info(f"Detected: {obj_type}")

    for detection in traffic_sign_detections.detections:
        category = detection.categories[0]
        if category.score > 0.6:
            distance = estimate_distance(detection.bounding_box.width, "stop sign")
            sign = category.category_name
            if distance:
                announcements.append(f"{sign} sign {distance} meters")
                logging.info(f"Detected: {sign} sign, Distance: {distance}m")
            else:
                announcements.append(f"{sign} sign")
                logging.info(f"Detected: {sign} sign")

    if user_behavior != "No user detected":
        announcements.append(f"You are {user_behavior}")

    if person_behavior[0] != "No person detected":
        announcements.append(f"Person ahead is {person_behavior[0]}")

    if text:
        announcements.append(f"Text detected: {text}")

    for obstacle, detected, distance in obstacles:
        if detected:
            if distance:
                announcements.append(f"{obstacle} {distance} meters")
                logging.info(f"Detected: {obstacle}, Distance: {distance}m")
            else:
                announcements.append(obstacle)
                logging.info(f"Detected: {obstacle}")

    if navigation_steps:
        announcements.append(navigation_steps[0])
        logging.info(f"Navigation: {navigation_steps[0]}")

    if announcements:
        announcement = ". ".join(announcements[:5])
        speak(announcement)
        last_announce_time = time.time()

def stream_to_caregiver(frame, overlays, alerts, user_behavior, location):
    """Stream data to caregiver"""
    _, buffer = cv2.imencode('.jpg', frame)
    data = {
        "frame": buffer.tobytes().hex(),
        "overlays": overlays,
        "alerts": alerts,
        "user_behavior": user_behavior,
        "location": location
    }
    sock.sendto(json.dumps(data).encode(), (CAREGIVER_IP, CAREGIVER_PORT))
    logging.info(f"Streamed to caregiver: Alerts: {alerts}, Location: {location}")

def settings_menu():
    """Voice-activated settings menu"""
    speak("Settings menu. Say a number from 1 to 3. 1: Navigate to a destination. 2: Update caregiver details. 3: Logout.")
    command = listen_for_command()
    if not command:
        speak("No command recognized.")
        return False

    if "1" in command or "one" in command:
        speak("Where do you wish to go?")
        destination = listen_for_command()
        if destination:
            user_data["destination"] = destination
            directions = get_directions(destination)
            if directions:
                speak(f"Destination set to {destination}. Distance is {directions['distance']} kilometers. Follow: {', '.join(directions['steps'])}")
                logging.info(f"Navigation set to {destination}, Distance: {directions['distance']}km")
                return directions
        speak("Could not find destination.")
        return False

    elif "2" in command or "two" in command:
        speak("Say the new caregiver phone number.")
        phone = listen_for_command()
        if phone:
            user_data["caregiver_phone"] = phone
            speak(f"Caregiver phone updated to {phone}")
            logging.info(f"Caregiver phone updated to {phone}")
        return False

    elif "3" in command or "three" in command:
        speak("Goodbye. Logging out.")
        logging.info("User logged out")
        return "logout"

    speak("Invalid option.")
    return False

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# Main loop
navigation_steps = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Update location
    user_data["location"] = geocoder.ip('me').latlng
    logging.info(f"Location updated: {user_data['location']}")

    # Detect obstacles
    speed_breaker_detected = detect_speed_breakers(frame)
    pothole_detected = detect_potholes(frame)
    curb_detected = detect_curbs(frame)
    stairs_detected = detect_stairs(frame)
    obstacles = [
        ("Speed breaker", speed_breaker_detected, estimate_distance(100, "pothole") if speed_breaker_detected else None),
        ("Pothole", pothole_detected, estimate_distance(100, "pothole") if pothole_detected else None),
        ("Curb", curb_detected, estimate_distance(50, "curb") if curb_detected else None),
        ("Stairs", stairs_detected, estimate_distance(200, "stairs") if stairs_detected else None)
    ]

    # Object detection
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = object_detector.detect(mp_image)
    traffic_sign_result = traffic_sign_detector.detect(mp_image)

    # Pose estimation
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(frame_rgb)
    user_behavior = "No user detected"
    person_behavior = ("No person detected", None)
    if pose_results.pose_landmarks:
        user_behavior, user_speed, user_gesture = analyze_user_behavior(
            pose_results.pose_landmarks, FRAME_WIDTH, FRAME_HEIGHT, user_positions, user_head_orientations, user_speeds
        )
        person_behavior = predict_person_behavior(pose_results.pose_landmarks, prev_person_positions, FRAME_WIDTH)
        if person_behavior[1] is not None:
            prev_person_positions.append(person_behavior[1])

    # Depth estimation
    depth = estimate_depth(pose_results.pose_landmarks, FRAME_WIDTH)

    # OCR
    text = read_text(frame)

    # Emergency detection
    for detection in detection_result.detections:
        category = detection.categories[0]
        distance = estimate_distance(detection.bounding_box.width, category.category_name)
        if distance and distance < 2 and category.score > 0.7:
            emergency_detected = True

    if emergency_detected:
        speak("Emergency detected. Call caregiver? Say yes or no.")
        command = listen_for_command()
        if command and "yes" in command:
            call_caregiver()
            speak("Calling caregiver")
        emergency_detected = False

    # AR overlays
    road_context = is_road_context(frame, bool(traffic_sign_result.detections), speed_breaker_detected)
    overlays = {"objects": [], "navigation": []}
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        category = detection.categories[0]
        label = classify_vehicle(category.category_name)
        distance = estimate_distance(bbox.width, category.category_name)
        overlays["objects"].append({
            "label": label,
            "score": round(category.score, 2),
            "bbox": [bbox.origin_x, bbox.origin_y, bbox.width, bbox.height],
            "distance": distance,
            "depth": depth
        })

    for detection in traffic_sign_result.detections:
        bbox = detection.bounding_box
        category = detection.categories[0]
        distance = estimate_distance(bbox.width, "stop sign")
        overlays["objects"].append({
            "label": category.category_name,
            "score": round(category.score, 2),
            "bbox": [bbox.origin_x, bbox.origin_y, bbox.width, bbox.height],
            "distance": distance,
            "depth": depth
        })

    for obstacle, detected, distance in obstacles:
        if detected:
            overlays["objects"].append({
                "label": obstacle,
                "score": 0.9,
                "bbox": [FRAME_WIDTH//4, FRAME_HEIGHT//4, FRAME_WIDTH//2, FRAME_HEIGHT//2],
                "distance": distance,
                "depth": depth
            })

    if navigation_steps and road_context:
        overlays["navigation"].append({"type": "arrow", "direction": "left", "x": 50, "y": 50})

    # Draw detections
    frame_with_detections = draw_object_detections(frame, detection_result, traffic_sign_result, user_behavior, navigation_steps, road_context)

    # Alerts
    alerts = []
    if speed_breaker_detected:
        alerts.append("Speed breaker ahead")
    if pothole_detected:
        alerts.append("Pothole ahead")
    if curb_detected:
        alerts.append("Curb ahead")
    if stairs_detected:
        alerts.append("Stairs ahead")
    if text:
        alerts.append(f"Text detected: {text}")
    if emergency_detected:
        alerts.append("Emergency: Object too close")

    # Announce detections
    announce_detections(detection_result, traffic_sign_result, user_behavior, person_behavior, text, obstacles, navigation_steps)

    # Stream to caregiver
    stream_to_caregiver(frame_with_detections, overlays, alerts, user_behavior, user_data["location"])

    # Process voice commands
    command = listen_for_command()
    if command:
        if "settings" in command:
            result = settings_menu()
            if result == "logout":
                break
            elif result:
                navigation_steps = result["steps"]
        elif "grocery store" in command:
            user_data["destination"] = "nearest grocery store"
            directions = get_directions("grocery store")
            if directions:
                speak(f"Navigating to grocery store. Distance is {directions['distance']} kilometers. Follow: {', '.join(directions['steps'])}")
                navigation_steps = directions["steps"]
                logging.info(f"Navigation to grocery store, Distance: {directions['distance']}km")
        elif "park" in command:
            user_data["destination"] = "nearest park"
            directions = get_directions("park")
            if directions:
                speak(f"Navigating to park. Distance is {directions['distance']} kilometers. Follow: {', '.join(directions['steps'])}")
                navigation_steps = directions["steps"]
                logging.info(f"Navigation to park, Distance: {directions['distance']}km")

    cv2.imshow('Advanced Obstacle Detection', frame_with_detections)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()
sock.close()