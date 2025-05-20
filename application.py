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
import uuid
from datetime import datetime
import logging
import urllib.request
import os
import base64
from flask import Flask, render_template
from flask_socketio import SocketIO
import threading
import redis
import json
import eventlet

# Patch eventlet
eventlet.monkey_patch()

# Initialize logging
logging.basicConfig(
    filename=f"caregiver_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize Flask and SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(
    app,
    ping_timeout=20,  # Increased for stability
    ping_interval=10,
    async_mode='eventlet',
    cors_allowed_origins=['http://localhost:5000', 'http://127.0.0.1:5000']
)

# ... (rest of your imports and code remain unchanged) ...

# Initialize Redis
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Initialize MediaPipe Object Detection
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

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Camera parameters
KNOWN_WIDTHS = {
    "car": 1.8, "truck": 2.5, "motorcycle": 0.8, "bicycle": 0.7,
    "person": 0.5, "tree": 1.0, "stop sign": 0.75, "bench": 1.5,
    "traffic light": 0.5, "pole": 0.3, "debris": 0.5, "barrier": 1.0,
    "pothole": 0.5, "curb": 0.3, "stairs": 1.0
}
FOCAL_LENGTH = 1000
FRAME_WIDTH = 640
FRAME_HEIGHT = 360

# User and caregiver data
user_data = {
    "user_id": str(uuid.uuid4()),
    "caregiver_phone": "caregiver_phone_number",
    "destination": None,
    "location": None,
    "behavior": "No user detected"
}

# Behavior tracking
user_positions = deque(maxlen=10)
user_head_orientations = deque(maxlen=5)
user_speeds = deque(maxlen=10)
prev_person_positions = deque(maxlen=10)
last_announce_time = time.time()
emergency_detected = False

# Optimization variables
last_ocr_time = time.time()
ocr_interval = 10
last_optical_flow_time = time.time()
optical_flow_interval = 0.5  # Increased for stability
last_location_update = time.time()
location_update_interval = 30
prev_gray = None
alerts = []
navigation_steps = []
prev_frame = None


# Download 3D object textures
def download_3d_objects():
    objects = {
        "arrow": "https://www.pngall.com/wp-content/uploads/2016/03/Arrow-PNG-Image.png",
        "cone": "https://www.pngall.com/wp-content/uploads/5/Traffic-Cone-PNG-Clipart.png",
        "marker": "https://www.pngall.com/wp-content/uploads/5/Dot-PNG-Image.png"
    }
    os.makedirs("3d_objects", exist_ok=True)
    for name, url in objects.items():
        if not os.path.exists(f"3d_objects/{name}.png"):
            try:
                urllib.request.urlretrieve(url, f"3d_objects/{name}.png")
                logging.info(f"Downloaded {name}.png")
            except Exception as e:
                logging.error(f"Failed to download {name}.png: {str(e)}")


download_3d_objects()
arrow_texture = cv2.imread("3d_objects/arrow.png", cv2.IMREAD_UNCHANGED)
cone_texture = cv2.imread("3d_objects/cone.png", cv2.IMREAD_UNCHANGED)
marker_texture = cv2.imread("3d_objects/marker.png", cv2.IMREAD_UNCHANGED)


def speak(text):
    try:
        engine.say(text)
        engine.runAndWait()
        logging.info(f"Spoken: {text}")
    except Exception as e:
        logging.error(f"Text-to-speech error: {str(e)}")


def listen_for_command():
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        speak("Listening...")
        try:
            audio = recognizer.listen(source, timeout=5)
            command = recognizer.recognize_google(audio).lower()
            logging.info(f"Voice command: {command}")
            return command
        except (sr.WaitTimeoutError, sr.UnknownValueError, sr.RequestError) as e:
            logging.info(f"No voice command recognized: {str(e)}")
            return None


def estimate_distance(object_width_pixels, object_type):
    if object_type not in KNOWN_WIDTHS:
        return None
    real_width = KNOWN_WIDTHS[object_type]
    distance = (real_width * FOCAL_LENGTH) / object_width_pixels if object_width_pixels > 0 else None
    return round(distance, 1) if distance else None


def estimate_depth(landmarks, frame_width):
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
    if label == "car":
        return "4-wheeler"
    elif label == "truck":
        return "6-wheeler"
    elif label in ["motorcycle", "bicycle"]:
        return "2-wheeler"
    return label


def analyze_user_behavior(pose_landmarks, frame_width, frame_height, prev_positions, prev_orientations, speeds):
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

    shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2 * frame_width
    nose_x = nose.x * frame_width
    head_orientation = "Looking forward"
    if nose_x < shoulder_mid_x - 25:
        head_orientation = "Looking left"
    elif nose_x > shoulder_mid_x + 25:
        head_orientation = "Looking right"
    prev_orientations.append(head_orientation)
    smoothed_orientation = max(set(prev_orientations), key=list(prev_orientations).count)

    hip_mid_x = (left_hip.x + right_hip.x) / 2 * frame_width
    hip_mid_y = (left_hip.y + right_hip.y) / 2 * frame_height
    movement = "Stationary"
    avg_speed = 0
    if prev_positions:
        prev_x, prev_y = prev_positions[-1]
        velocity_x = hip_mid_x - prev_x
        velocity_y = hip_mid_y - prev_y
        speed = np.sqrt(velocity_x ** 2 + velocity_y ** 2) / frame_width * 100
        speeds.append(speed)
        avg_speed = np.mean(speeds)

        if avg_speed > 5:
            if abs(velocity_x) > abs(velocity_y):
                movement = "Turning left" if velocity_x < -5 else "Turning right"
            else:
                movement = "Moving forward" if velocity_y < -5 else "Moving backward"
        elif avg_speed > 2:
            movement = "Slowing down"
        else:
            movement = "Stationary"

    prev_positions.append((hip_mid_x, hip_mid_y))

    gesture = None
    shoulder_height = (left_shoulder.y + right_shoulder.y) / 2 * frame_height
    if left_wrist.y * frame_height < shoulder_height - 25 and left_wrist.visibility > 0.7:
        gesture = "Waving left hand"
    elif right_wrist.y * frame_height < shoulder_height - 25 and right_wrist.visibility > 0.7:
        gesture = "Waving right hand"
    elif abs(left_wrist.x - right_wrist.x) * frame_width > 50 and left_wrist.y * frame_height < shoulder_height:
        gesture = "Pointing"

    hip_height = (left_hip.y + right_hip.y) / 2 * frame_height
    knee_height = (left_knee.y + right_knee.y) / 2 * frame_height
    crouching = "Crouching" if knee_height < hip_height + 25 and avg_speed < 2 else None

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
    if not pose_landmarks:
        return "No person detected", None
    left_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    hip_mid_x = (left_hip.x + right_hip.x) / 2 * frame_width
    if prev_positions:
        prev_x = prev_positions[-1]
        velocity = hip_mid_x - prev_x
        if velocity > 5:
            return "Moving right", hip_mid_x
        elif velocity < -5:
            return "Moving left", hip_mid_x
        return "Stationary", hip_mid_x
    return "Stationary", hip_mid_x


def detect_speed_breakers(frame):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                                minLineLength=50, maxLineGap=10)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y1 - y2) < 10 and abs(x1 - x2) > 50:
                    return True
        return False
    except Exception as e:
        logging.error(f"Speed breaker detection error: {str(e)}")
        return False


def detect_potholes(frame):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 500:
                return True
        return False
    except Exception as e:
        logging.error(f"Pothole detection error: {str(e)}")
        return False


def detect_curbs(frame):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                                minLineLength=25, maxLineGap=5)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y1 - y2) > 25 and abs(x1 - x2) < 10:
                    return True
        return False
    except Exception as e:
        logging.error(f"Curb detection error: {str(e)}")
        return False


def detect_stairs(frame):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                                minLineLength=40, maxLineGap=10)
        if lines is not None:
            horizontal_lines = sum(1 for line in lines if abs(line[0][1] - line[0][3]) < 10)
            if horizontal_lines > 3:
                return True
        return False
    except Exception as e:
        logging.error(f"Stairs detection error: {str(e)}")
        return False


def is_road_context(frame, speed_breaker):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                                minLineLength=50, maxLineGap=10)
        line_count = len(lines) if lines is not None else 0
        return line_count > 5 or speed_breaker
    except Exception as e:
        logging.error(f"Road context detection error: {str(e)}")
        return False


def read_text(frame):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray, config='--psm 6')
        text = text.strip() if text.strip() else None
        if text:
            logging.info(f"OCR text: {text}")
        return text
    except Exception as e:
        logging.error(f"OCR error: {str(e)}")
        return None


def get_directions(destination):
    try:
        g = geocoder.osm(destination)
        if g.ok:
            user_loc = geocoder.ip('me').latlng
            dest_loc = g.latlng
            distance = math.sqrt((dest_loc[0] - user_loc[0]) ** 2 + (dest_loc[1] - user_loc[1]) ** 2) * 111
            return {"distance": round(distance, 1),
                    "steps": ["Turn left in 50 meters", "Continue straight for 200 meters"]}
        return None
    except Exception as e:
        logging.error(f"Geocoder error: {str(e)}")
        return None


def call_caregiver():
    try:
        speak("Calling caregiver (mock implementation)")
        logging.info("Caregiver call initiated (mock)")
        return True
    except Exception as e:
        logging.error(f"Caregiver call error: {str(e)}")
        return False


def analyze_crowd_density(detections):
    try:
        person_count = sum(
            1 for detection in detections.detections if detection.categories[0].category_name == "person")
        avg_distance = 0
        if person_count > 0:
            distances = [
                estimate_distance(detection.bounding_box.width, "person")
                for detection in detections.detections
                if detection.categories[0].category_name == "person" and estimate_distance(detection.bounding_box.width,
                                                                                           "person") is not None
            ]
            avg_distance = np.mean(distances) if distances else 0

        if person_count >= 5 or (person_count > 0 and avg_distance < 3):
            density = "High"
            warning = "Crowded area ahead, slow down"
        elif person_count >= 3 or (person_count > 0 and avg_distance < 5):
            density = "Medium"
            warning = "Moderately crowded area, proceed with caution"
        else:
            density = "Low"
            warning = None

        logging.info(f"Crowd density: {density}, Person count: {person_count}, Avg distance: {avg_distance:.1f}m")
        return density, warning, person_count, avg_distance
    except Exception as e:
        logging.error(f"Crowd density analysis error: {str(e)}")
        return "Low", None, 0, 0


def compute_optical_flow(prev_frame, curr_frame):
    global prev_gray
    if prev_frame is None:
        return None, None

    try:
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        avg_mag = np.mean(mag)

        moving_objects = []
        if avg_mag > 1.0:
            h, w = curr_frame.shape[:2]
            for detection in object_detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                                             data=cv2.cvtColor(curr_frame,
                                                                               cv2.COLOR_BGR2RGB))).detections:
                bbox = detection.bounding_box
                label = detection.categories[0].category_name
                if label in ["person", "car", "truck", "motorcycle", "bicycle"]:
                    x, y, w_b, h_b = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(w, x + w_b), min(h, y + h_b)
                    roi_flow = flow[y1:y2, x1:x2]
                    if roi_flow.size > 0:
                        roi_mag, roi_ang = cv2.cartToPolar(roi_flow[..., 0], roi_flow[..., 1])
                        avg_roi_mag = np.mean(roi_mag)
                        avg_roi_ang = np.mean(roi_ang) * 180 / np.pi
                        if avg_roi_mag > 1.5:
                            direction = "left" if 45 < avg_roi_ang < 135 else "right" if 225 < avg_roi_ang < 315 else "towards you"
                            moving_objects.append((label, direction, avg_roi_mag, (x + w_b // 2, y + h_b // 2)))

        return moving_objects, flow
    except Exception as e:
        logging.error(f"Optical flow error: {str(e)}")
        return None, None


def draw_object_detections(frame, detections, user_behavior, navigation_steps, road_context, crowd_density,
                           moving_objects, flow):
    global prev_gray
    try:
        annotated_image = frame.copy()
        height, width = frame.shape[:2]

        if road_context:
            for detection in detections.detections:
                bbox = detection.bounding_box
                category = detection.categories[0]
                label = classify_vehicle(category.category_name)
                score = round(category.score, 2)
                distance = estimate_distance(bbox.width, category.category_name)

                color = (0, 255, 0) if distance and distance > 5 else (0, 0, 255)
                cv2.rectangle(annotated_image, (bbox.origin_x, bbox.origin_y),
                              (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height), color, 1)
                label_text = f"{label} ({score})"
                if distance:
                    label_text += f" {distance}m"
                cv2.putText(annotated_image, label_text, (bbox.origin_x, bbox.origin_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            if navigation_steps:
                cv2.arrowedLine(annotated_image, (25, 25), (50, 25), (0, 255, 255), 2)
                cv2.putText(annotated_image, navigation_steps[0], (25, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            if crowd_density:
                density_text = f"Crowd: {crowd_density[0]} ({crowd_density[2]} people)"
                if crowd_density[3] > 0:
                    density_text += f", Avg {crowd_density[3]:.1f}m"
                cv2.putText(annotated_image, density_text, (5, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)

            if moving_objects:
                for label, direction, mag, center in moving_objects:
                    arrow_length = int(mag * 10)
                    angle = 0 if direction == "right" else 180 if direction == "left" else 90
                    rad = math.radians(angle)
                    end_x = int(center[0] + arrow_length * math.cos(rad))
                    end_y = int(center[1] - arrow_length * math.sin(rad))
                    cv2.arrowedLine(annotated_image, center, (end_x, end_y), (255, 0, 255), 1)
                    cv2.putText(annotated_image, f"{label} {direction}", (center[0], center[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            if flow is not None:
                hsv = np.zeros_like(frame)
                hsv[..., 1] = 255
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hsv[..., 0] = ang * 180 / np.pi / 2
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                annotated_image = cv2.addWeighted(annotated_image, 0.8, flow_bgr, 0.2, 0)

        cv2.putText(annotated_image, f"User: {user_behavior}", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return annotated_image
    except Exception as e:
        logging.error(f"Draw detections error: {str(e)}")
        return frame


def announce_detections(detections, user_behavior, person_behavior, text, obstacles, navigation_steps, crowd_density,
                        moving_objects):
    global last_announce_time
    if time.time() - last_announce_time < 5:
        return

    try:
        person_count = crowd_density[2] if crowd_density else 0
        close_object = False
        for detection in detections.detections:
            category = detection.categories[0]
            distance = estimate_distance(detection.bounding_box.width, category.category_name)
            if distance and distance < 0.5 and category.score > 0.7:
                close_object = True
                break

        if not (close_object or person_count > 5 or text):
            return

        announcements = []
        for detection in detections.detections:
            category = detection.categories[0]
            if category.score > 0.7:
                obj_type = classify_vehicle(category.category_name)
                distance = estimate_distance(detection.bounding_box.width, category.category_name)
                if distance and distance < 0.5:
                    announcements.append(f"{obj_type} at {distance} meters")
                    logging.info(f"Detected: {obj_type}, Distance: {distance}m")

        if text:
            announcements.append(f"Text detected: {text}")

        if announcements:
            announcement = ". ".join(announcements[:5])
            speak(announcement)
            last_announce_time = time.time()
    except Exception as e:
        logging.error(f"Announce detections error: {str(e)}")


def settings_menu():
    try:
        speak(
            "Settings menu. Say a number from 1 to 3. 1: Navigate to a destination. 2: Update caregiver details. 3: Logout.")
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
                    speak(
                        f"Destination set to {destination}. Distance is {directions['distance']} kilometers. Follow: {', '.join(directions['steps'])}")
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
    except Exception as e:
        logging.error(f"Settings menu error: {str(e)}")
        return False


def overlay_3d_object(frame, texture, position, size=50):
    if texture is None:
        return frame

    try:
        texture = cv2.resize(texture, (size, size))
        if texture.shape[2] == 4:
            alpha = texture[:, :, 3] / 255.0
            texture = texture[:, :, :3]
        else:
            alpha = np.ones((size, size)) * 0.7

        x, y = position
        y1, y2 = max(0, y - size // 2), min(frame.shape[0], y + size // 2)
        x1, x2 = max(0, x - size // 2), min(frame.shape[1], x + size // 2)

        if y2 - y1 > 0 and x2 - x1 > 0:
            texture_roi = texture[0:(y2 - y1), 0:(x2 - x1)]
            alpha_roi = alpha[0:(y2 - y1), 0:(x2 - x1)]
            for c in range(3):
                frame[y1:y2, x1:x2, c] = (
                        alpha_roi * texture_roi[:, :, c] +
                        (1 - alpha_roi) * frame[y1:y2, x1:x2, c]
                )
        return frame
    except Exception as e:
        logging.error(f"Overlay 3D object error: {str(e)}")
        return frame


def detect_lanes(frame):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)

        if lines is None:
            return False

        left_lanes = 0
        right_lanes = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if 20 < angle < 70:
                left_lanes += 1
            elif -70 < angle < -20:
                right_lanes += 1

        return left_lanes >= 1 and right_lanes >= 1
    except Exception as e:
        logging.error(f"Lane detection error: {str(e)}")
        return False


@app.route('/')
def index():
    return render_template('ar_.html')


@socketio.on('connect', namespace='/video_feed')
def handle_connect():
    logging.info('WebSocket client connected')
    socketio.emit('status', {'message': 'Connected'}, namespace='/video_feed')


@socketio.on('disconnect', namespace='/video_feed')
def handle_disconnect():
    logging.info('WebSocket client disconnected')


@socketio.on_error(namespace='/video_feed')
def handle_error(e):
    logging.error(f"WebSocket error: {str(e)}")


@socketio.on('message', namespace='/video_feed')
def handle_frame(data):
    global last_ocr_time, last_optical_flow_time, last_location_update, prev_gray, alerts, navigation_steps, emergency_detected, prev_frame
    try:
        # Decode base64 frame
        img_data = base64.b64decode(data.split(',')[1])
        npimg = np.frombuffer(img_data, dtype=np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # Update location
        if time.time() - last_location_update > location_update_interval:
            try:
                user_data["location"] = geocoder.ip('me').latlng
                logging.info(f"Location updated: {user_data['location']}")
                last_location_update = time.time()
            except Exception as e:
                logging.error(f"Location update error: {str(e)}")

        # Detect obstacles
        speed_breaker_detected = detect_speed_breakers(frame)
        pothole_detected = detect_potholes(frame)
        curb_detected = detect_curbs(frame)
        stairs_detected = detect_stairs(frame)
        obstacles = [
            ("Speed breaker", speed_breaker_detected,
             estimate_distance(100, "pothole") if speed_breaker_detected else None),
            ("Pothole", pothole_detected, estimate_distance(100, "pothole") if pothole_detected else None),
            ("Curb", curb_detected, estimate_distance(50, "curb") if curb_detected else None),
            ("Stairs", stairs_detected, estimate_distance(200, "stairs") if stairs_detected else None)
        ]

        # Object detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = object_detector.detect(mp_image)

        # Crowd density analysis
        crowd_density = analyze_crowd_density(detection_result)

        # Optical flow analysis
        moving_objects, flow = None, None
        if time.time() - last_optical_flow_time > optical_flow_interval and prev_frame is not None:
            moving_objects, flow = compute_optical_flow(prev_frame, frame)
            last_optical_flow_time = time.time()

        # Pose estimation
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(frame_rgb)
        user_behavior = "No user detected"
        person_behavior = ("No person detected", None)
        if pose_results.pose_landmarks:
            user_behavior, user_speed, user_gesture = analyze_user_behavior(
                pose_results.pose_landmarks, FRAME_WIDTH, FRAME_HEIGHT, user_positions, user_head_orientations,
                user_speeds
            )
            person_behavior = predict_person_behavior(pose_results.pose_landmarks, prev_person_positions, FRAME_WIDTH)
            if person_behavior[1] is not None:
                prev_person_positions.append(person_behavior[1])

        # Depth estimation
        depth = estimate_depth(pose_results.pose_landmarks, FRAME_WIDTH)

        # OCR
        text = None
        if time.time() - last_ocr_time > ocr_interval:
            text = read_text(frame)
            last_ocr_time = time.time()

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

        # Detect road lanes
        is_two_lane_road = detect_lanes(frame)
        if is_two_lane_road:
            frame = overlay_3d_object(frame, cone_texture, (frame.shape[1] // 2, frame.shape[0] - 50), size=60)

        # AR overlays
        road_context = is_road_context(frame, speed_breaker_detected)
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

        for obstacle, detected, distance in obstacles:
            if detected:
                overlays["objects"].append({
                    "label": obstacle,
                    "score": 0.9,
                    "bbox": [FRAME_WIDTH // 4, FRAME_HEIGHT // 4, FRAME_WIDTH // 2, FRAME_HEIGHT // 2],
                    "distance": distance,
                    "depth": depth
                })

        if navigation_steps and road_context:
            overlays["navigation"].append({"type": "arrow", "direction": "left", "x": 25, "y": 25})
            frame = overlay_3d_object(frame, arrow_texture, (25, 25), size=75)

        # Draw detections
        frame_with_detections = draw_object_detections(frame, detection_result, user_behavior, navigation_steps,
                                                       road_context, crowd_density, moving_objects, flow)

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
        if crowd_density[1]:
            alerts.append(crowd_density[1])
        if moving_objects:
            for label, direction, _, _ in moving_objects:
                alerts.append(f"{label} approaching from the {direction}")

        # Announce detections
        announce_detections(detection_result, user_behavior, person_behavior, text, obstacles, navigation_steps,
                            crowd_density, moving_objects)

        # Save to Redis
        user_data["behavior"] = user_behavior
        redis_client.set('user_data', json.dumps(user_data))
        redis_client.set('alerts', json.dumps(alerts))
        _, buffer = cv2.imencode('.jpg', frame_with_detections)
        redis_client.set('frame', base64.b64encode(buffer).decode('utf-8'))

        # Encode frame to base64
        _, buffer = cv2.imencode('.jpg', frame_with_detections, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')

        # Send to client
        socketio.emit('frame', {'image': frame_base64, 'alerts': alerts}, namespace='/video_feed')

        # Update previous frame
        prev_frame = frame.copy()

        # Process voice commands in a separate thread
        def process_voice_commands():
            command = listen_for_command()
            if command:
                if "settings" in command:
                    result = settings_menu()
                    if result == "logout":
                        socketio.emit('logout', namespace='/video_feed')
                    elif result:
                        global navigation_steps
                        navigation_steps = result["steps"]
                elif "grocery store" in command:
                    user_data["destination"] = "nearest grocery store"
                    directions = get_directions("grocery store")
                    if directions:
                        speak(
                            f"Navigating to grocery store. Distance is {directions['distance']} kilometers. Follow: {', '.join(directions['steps'])}")
                        navigation_steps = directions["steps"]
                        logging.info(f"Navigation to grocery store, Distance: {directions['distance']}km")
                elif "park" in command:
                    user_data["destination"] = "nearest park"
                    directions = get_directions("park")
                    if directions:
                        speak(
                            f"Navigating to park. Distance is {directions['distance']} kilometers. Follow: {', '.join(directions['steps'])}")
                        navigation_steps = directions["steps"]
                        logging.info(f"Navigation to park, Distance: {directions['distance']}km")

        threading.Thread(target=process_voice_commands, daemon=True).start()

    except Exception as e:
        logging.error(f"Frame processing error: {str(e)}")
        socketio.emit('error', {'message': 'Frame processing failed'}, namespace='/video_feed')


if __name__ == '__main__':
    try:
        socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
    finally:
        pose.close()