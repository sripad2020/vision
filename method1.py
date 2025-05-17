import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyttsx3
import time
import numpy as np
import sqlite3
import uuid
import pytesseract
from datetime import datetime
import threading
import queue
import os
import speech_recognition as sr

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)


# Database setup
def init_db():
    conn = sqlite3.connect('assistive_vision.db')
    cursor = conn.cursor()

    # Create tables if they don't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS caregivers (
        id TEXT PRIMARY KEY,
        username TEXT UNIQUE,
        password TEXT,
        name TEXT,
        email TEXT,
        phone TEXT
    )''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        username TEXT UNIQUE,
        password TEXT,
        name TEXT,
        email TEXT,
        phone TEXT,
        caregiver_id TEXT,
        preferences TEXT
    )''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sessions (
        user_id TEXT,
        session_token TEXT,
        expiry_time TEXT,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )''')

    conn.commit()
    conn.close()


init_db()


# User Management Classes
class User:
    def __init__(self, user_id, username, name, role):
        self.user_id = user_id
        self.username = username
        self.name = name
        self.role = role  # 'caregiver' or 'user'
        self.session_token = str(uuid.uuid4())
        self.save_session()

    def save_session(self):
        conn = sqlite3.connect('assistive_vision.db')
        cursor = conn.cursor()
        expiry = datetime.now().timestamp() + 3600  # 1 hour expiry
        cursor.execute('INSERT INTO sessions VALUES (?, ?, ?)',
                       (self.user_id, self.session_token, str(expiry)))
        conn.commit()
        conn.close()

    def logout(self):
        conn = sqlite3.connect('assistive_vision.db')
        cursor = conn.cursor()
        cursor.execute('DELETE FROM sessions WHERE user_id=?', (self.user_id,))
        conn.commit()
        conn.close()


class Caregiver(User):
    def __init__(self, user_id, username, name):
        super().__init__(user_id, username, name, 'caregiver')
        self.linked_users = []
        self.load_linked_users()

    def load_linked_users(self):
        conn = sqlite3.connect('assistive_vision.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id, name FROM users WHERE caregiver_id=?', (self.user_id,))
        self.linked_users = [{'id': row[0], 'name': row[1]} for row in cursor.fetchall()]
        conn.close()


class VisuallyImpairedUser(User):
    def __init__(self, user_id, username, name, caregiver_id=None, preferences=None):
        super().__init__(user_id, username, name, 'user')
        self.caregiver_id = caregiver_id
        self.preferences = preferences or {
            'voice_speed': 150,
            'detection_threshold': 0.5,
            'announcement_frequency': 5
        }
        self.camera_feed_queue = queue.Queue()
        self.alerts = []

    def update_preferences(self, new_prefs):
        self.preferences.update(new_prefs)
        conn = sqlite3.connect('assistive_vision.db')
        cursor = conn.cursor()
        cursor.execute('UPDATE users SET preferences=? WHERE id=?',
                       (str(self.preferences), self.user_id))
        conn.commit()
        conn.close()
        engine.setProperty('rate', self.preferences['voice_speed'])


# Authentication System
class AuthSystem:
    @staticmethod
    def register_caregiver(username, password, name, email, phone):
        conn = sqlite3.connect('assistive_vision.db')
        cursor = conn.cursor()

        # Check if username exists
        cursor.execute('SELECT username FROM caregivers WHERE username=?', (username,))
        if cursor.fetchone():
            conn.close()
            return None

        caregiver_id = 'cg-' + str(uuid.uuid4())
        cursor.execute('INSERT INTO caregivers VALUES (?, ?, ?, ?, ?, ?)',
                       (caregiver_id, username, password, name, email, phone))
        conn.commit()
        conn.close()
        return Caregiver(caregiver_id, username, name)

    @staticmethod
    def register_user(username, password, name, email, phone, caregiver_id=None):
        conn = sqlite3.connect('assistive_vision.db')
        cursor = conn.cursor()

        # Check if username exists
        cursor.execute('SELECT username FROM users WHERE username=?', (username,))
        if cursor.fetchone():
            conn.close()
            return None

        user_id = 'usr-' + str(uuid.uuid4())
        preferences = str({'voice_speed': 150, 'detection_threshold': 0.5, 'announcement_frequency': 5})
        cursor.execute('INSERT INTO users VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                       (user_id, username, password, name, email, phone, caregiver_id, preferences))
        conn.commit()
        conn.close()
        return VisuallyImpairedUser(user_id, username, name, caregiver_id)

    @staticmethod
    def login(username, password, role):
        conn = sqlite3.connect('assistive_vision.db')
        cursor = conn.cursor()

        table = 'caregivers' if role == 'caregiver' else 'users'
        cursor.execute(f'SELECT id, name FROM {table} WHERE username=? AND password=?',
                       (username, password))
        result = cursor.fetchone()
        conn.close()

        if not result:
            return None

        user_id, name = result
        if role == 'caregiver':
            return Caregiver(user_id, username, name)
        else:
            cursor.execute('SELECT caregiver_id, preferences FROM users WHERE id=?', (user_id,))
            caregiver_id, prefs = cursor.fetchone()
            preferences = eval(prefs) if prefs else None
            return VisuallyImpairedUser(user_id, username, name, caregiver_id, preferences)

    @staticmethod
    def voice_authentication():
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            engine.say("Please say your username and password separated by the word 'password'")
            engine.runAndWait()
            audio = recognizer.listen(source, timeout=10)

            try:
                text = recognizer.recognize_google(audio)
                parts = text.lower().split('password')
                if len(parts) == 2:
                    username = parts[0].strip()
                    password = parts[1].strip()
                    return username, password
            except Exception as e:
                print("Voice recognition error:", e)
            return None, None


# OCR Functionality
def perform_ocr(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text.strip()


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


def draw_object_detections(frame, detections, user_prefs):
    annotated_image = frame.copy()
    height, width = frame.shape[:2]

    for detection in detections.detections:
        bbox = detection.bounding_box
        category = detection.categories[0]

        if category.score < user_prefs['detection_threshold']:
            continue

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


def announce_detections(detections, user_prefs, last_announce_time):
    current_time = time.time()
    if current_time - last_announce_time < user_prefs['announcement_frequency']:
        return last_announce_time

    if detections:
        objects = []
        for detection in detections.detections:
            category = detection.categories[0]
            if category.score > user_prefs['detection_threshold']:
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
            return current_time

    return last_announce_time


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


def user_camera_loop(user, stop_event):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    last_announce_time = time.time()
    last_ocr_time = time.time()

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue

        # OCR every 10 seconds
        current_time = time.time()
        if current_time - last_ocr_time > 10:
            text = perform_ocr(frame)
            if text:
                engine.say("I see text: " + text[:100])  # Limit to 100 chars
                engine.runAndWait()
            last_ocr_time = current_time

        # Object detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = object_detector.detect(mp_image)

        # Speed breaker detection
        speed_breaker_detected = detect_speed_breakers(frame)
        if speed_breaker_detected and current_time - last_announce_time > 1:
            engine.say("Speed breaker ahead")
            engine.runAndWait()
            last_announce_time = current_time

        # Announce detections
        last_announce_time = announce_detections(detection_result, user.preferences, last_announce_time)

        # Put frame in queue for caregive   r
        if user.camera_feed_queue.qsize() < 3:  # Keep queue size manageable
            user.camera_feed_queue.put(frame.copy())

    cap.release()


def caregiver_dashboard(caregiver):
    stop_events = {}

    def view_user_feed(user_id):
        stop_event = threading.Event()
        stop_events[user_id] = stop_event

        while not stop_event.is_set():
            user = next((u for u in caregiver.linked_users if u['id'] == user_id), None)
            if not user:
                break

            if caregiver.linked_users[0]['camera_feed_queue'].empty():  # Simplified
                continue

            frame = caregiver.linked_users[0]['camera_feed_queue'].get()
            cv2.imshow(f"User {user['name']} Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    print("\nCaregiver Dashboard")
    print("-------------------")
    print(f"Welcome, {caregiver.name}!")
    print(f"Linked users: {len(caregiver.linked_users)}")

    for i, user in enumerate(caregiver.linked_users, 1):
        print(f"{i}. {user['name']}")

    print("\nOptions:")
    print("1. View user feed")
    print("2. Refresh user list")
    print("3. Logout")

    choice = input("Select an option: ")
    if choice == '1' and caregiver.linked_users:
        user_idx = int(input("Select user number: ")) - 1
        if 0 <= user_idx < len(caregiver.linked_users):
            view_user_feed(caregiver.linked_users[user_idx]['id'])
    elif choice == '2':
        caregiver.load_linked_users()
    elif choice == '3':
        for event in stop_events.values():
            event.set()
        caregiver.logout()
        return False

    return True


def user_interface(user):
    stop_event = threading.Event()
    camera_thread = threading.Thread(target=user_camera_loop, args=(user, stop_event))
    camera_thread.start()

    print("\nUser Interface")
    print("-------------")
    print(f"Welcome, {user.name}!")
    print("\nOptions:")
    print("1. Adjust preferences")
    print("2. Link/unlink caregiver")
    print("3. Logout")

    choice = input("Select an option: ")
    if choice == '1':
        print("\nCurrent preferences:")
        for k, v in user.preferences.items():
            print(f"{k}: {v}")

        new_prefs = {}
        speed = input(f"Voice speed (current: {user.preferences['voice_speed']}): ")
        if speed:
            new_prefs['voice_speed'] = int(speed)
        threshold = input(f"Detection threshold (current: {user.preferences['detection_threshold']}): ")
        if threshold:
            new_prefs['detection_threshold'] = float(threshold)
        freq = input(f"Announcement frequency (current: {user.preferences['announcement_frequency']}): ")
        if freq:
            new_prefs['announcement_frequency'] = int(freq)

        if new_prefs:
            user.update_preferences(new_prefs)
            print("Preferences updated!")

    elif choice == '2':
        conn = sqlite3.connect('assistive_vision.db')
        cursor = conn.cursor()

        if user.caregiver_id:
            cursor.execute('SELECT name FROM caregivers WHERE id=?', (user.caregiver_id,))
            cg_name = cursor.fetchone()[0]
            print(f"\nCurrently linked to caregiver: {cg_name}")
            unlink = input("Unlink this caregiver? (y/n): ")
            if unlink.lower() == 'y':
                user.caregiver_id = None
                cursor.execute('UPDATE users SET caregiver_id=NULL WHERE id=?', (user.user_id,))
                conn.commit()
                print("Caregiver unlinked!")
        else:
            cursor.execute('SELECT id, name FROM caregivers')
            caregivers = cursor.fetchall()
            if caregivers:
                print("\nAvailable caregivers:")
                for i, (cg_id, cg_name) in enumerate(caregivers, 1):
                    print(f"{i}. {cg_name}")

                choice = input("Select caregiver to link (or 0 to cancel): ")
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(caregivers):
                        user.caregiver_id = caregivers[idx][0]
                        cursor.execute('UPDATE users SET caregiver_id=? WHERE id=?',
                                       (user.caregiver_id, user.user_id))
                        conn.commit()
                        print(f"Linked to caregiver: {caregivers[idx][1]}")
            else:
                print("No caregivers available")

        conn.close()

    elif choice == '3':
        stop_event.set()
        camera_thread.join()
        user.logout()
        return False

    return True


def main_menu():
    print("\nAssistive Vision System")
    print("----------------------")
    print("1. Register as Caregiver")
    print("2. Register as User")
    print("3. Login as Caregiver")
    print("4. Login as User")
    print("5. Voice Authentication")
    print("6. Exit")

    choice = input("Select an option: ")

    if choice == '1':
        print("\nCaregiver Registration")
        username = input("Username: ")
        password = input("Password: ")
        name = input("Full name: ")
        email = input("Email: ")
        phone = input("Phone: ")

        caregiver = AuthSystem.register_caregiver(username, password, name, email, phone)
        if caregiver:
            print(f"\nRegistration successful! Your caregiver ID is: {caregiver.user_id}")
            engine.say(f"Welcome {name}. You are now registered as a caregiver.")
            engine.runAndWait()
        else:
            print("Registration failed. Username may already exist.")

    elif choice == '2':
        print("\nUser Registration")
        username = input("Username: ")
        password = input("Password: ")
        name = input("Full name: ")
        email = input("Email: ")
        phone = input("Phone: ")
        caregiver_id = input("Caregiver ID (optional): ")

        user = AuthSystem.register_user(username, password, name, email, phone, caregiver_id)
        if user:
            print(f"\nRegistration successful! Your user ID is: {user.user_id}")
            engine.say(f"Welcome {name}. You are now registered as a user.")
            engine.runAndWait()
        else:
            print("Registration failed. Username may already exist.")

    elif choice == '3':
        print("\nCaregiver Login")
        username = input("Username: ")
        password = input("Password: ")

        caregiver = AuthSystem.login(username, password, 'caregiver')
        if caregiver:
            print(f"\nLogin successful! Welcome, {caregiver.name}.")
            engine.say(f"Welcome back {caregiver.name}")
            engine.runAndWait()

            while caregiver_dashboard(caregiver):
                pass
        else:
            print("Login failed. Invalid credentials.")

    elif choice == '4':
        print("\nUser Login")
        username = input("Username: ")
        password = input("Password: ")

        user = AuthSystem.login(username, password, 'user')
        if user:
            print(f"\nLogin successful! Welcome, {user.name}.")
            engine.say(f"Welcome back {user.name}")
            engine.runAndWait()

            while user_interface(user):
                pass
        else:
            print("Login failed. Invalid credentials.")

    elif choice == '5':
        print("\nVoice Authentication")
        username, password = AuthSystem.voice_authentication()
        if username and password:
            print("\nAttempting login...")

            # Try both roles
            user = AuthSystem.login(username, password, 'user')
            if user:
                print(f"\nLogin successful! Welcome, {user.name}.")
                engine.say(f"Welcome back {user.name}")
                engine.runAndWait()

                while user_interface(user):
                    pass
            else:
                caregiver = AuthSystem.login(username, password, 'caregiver')
                if caregiver:
                    print(f"\nLogin successful! Welcome, {caregiver.name}.")
                    engine.say(f"Welcome back {caregiver.name}")
                    engine.runAndWait()

                    while caregiver_dashboard(caregiver):
                        pass
                else:
                    print("Login failed. Invalid credentials.")
        else:
            print("Could not recognize your voice credentials.")

    elif choice == '6':
        return False

    return True

if __name__ == "__main__":
    while main_menu():
        pass
    print("Goodbye!")