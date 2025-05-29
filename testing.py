import eventlet
eventlet.monkey_patch()
import mediapipe as mp
import base64
import cv2
import numpy as np
import logging
import time
import threading
import uuid
import urllib.request
import os
import geocoder
import math
import re
import pytesseract
import glob
from datetime import datetime   
from collections import deque
from flask import Flask, render_template, jsonify, request, session, flash, redirect, url_for,send_file
from flask_socketio import SocketIO, emit
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import speech_recognition as sr
import pyttsx3
import sqlite3
import secrets
from werkzeug.security import generate_password_hash, check_password_hash
import socket
import colorsys
import networkx as nx
import pyaudio
import tensorflow as tf
import tensorflow_hub as hub
from scipy.spatial.distance import cdist

# Configure Tesseract path (update for your system)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Configure logging for each caregiver-caretaker pair
def configure_logging(caregiver_username, caretaker_username):
    log_filename = f"logs/caregiver_{caregiver_username}_caretaker_{caretaker_username}.log"
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    logger = logging.getLogger(f'CaregiverLogger_{caregiver_username}_{caretaker_username}')
    logger.setLevel(logging.INFO)
    if not logger.handlers:  # Prevent adding duplicate handlers
        file_handler = logging.FileHandler(log_filename, mode='a')
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        logger.handlers = []
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    logger.info(f"Logger initialized for {log_filename}")
    return logger

app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(16)
socketio = SocketIO(
    app,
    ping_timeout=20,
    ping_interval=10,
    async_mode='eventlet',
    cors_allowed_origins='*'
)

DATABASE = 'users.db'

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception as e:
        logger = logging.getLogger('CaregiverLogger')
        logger.error(f"Failed to get local IP: {str(e)}")
        return "Unknown"

def init_db():
    conn = sqlite3.connect(DATABASE, check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'caregiver'
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS users_info (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        speech_credential TEXT NOT NULL,
        caregiver_username TEXT NOT NULL,
        FOREIGN KEY (caregiver_username) REFERENCES users(username)
    )''')
    conn.commit()
    conn.close()

def get_db_connection():
    conn = sqlite3.connect(DATABASE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

init_db()

# Speech synthesis system
engine = None
speech_lock = threading.Lock()
speech_queue = []
is_speaking = False

def init_engine():
    global engine
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        voices = engine.getProperty('voices')
        if len(voices) > 1:
            engine.setProperty('voice', voices[1].id)
        logger = logging.getLogger('CaregiverLogger')
        logger.info("Speech engine initialized")
    except Exception as e:
        logger = logging.getLogger('CaregiverLogger')
        logger.error(f"Failed to initialize speech engine: {str(e)}")
        engine = None

def speak(text, priority=False):
    global is_speaking
    if not text or not isinstance(text, str):
        return
    with speech_lock:
        if priority:
            speech_queue.insert(0, text)
        else:
            speech_queue.append(text)
    if not is_speaking:
        threading.Thread(target=_process_speech_queue, daemon=True).start()

def _process_speech_queue():
    global is_speaking, engine
    if engine is None:
        init_engine()
        if engine is None:
            return
    while True:
        with speech_lock:
            if not speech_queue:
                is_speaking = False
                return
            text = speech_queue.pop(0)
            is_speaking = True
        try:
            engine.say(text)
            engine.runAndWait()
            time.sleep(0.1)
        except RuntimeError as e:
            if 'run loop already started' in str(e):
                logger = logging.getLogger('CaregiverLogger')
                logger.warning("Speech engine busy, retrying...")
                with speech_lock:
                    speech_queue.insert(0, text)
                time.sleep(0.5)
                continue
            logger = logging.getLogger('CaregiverLogger')
            logger.error(f"Speech error: {str(e)}")
            engine = None
        except Exception as e:
            logger = logging.getLogger('CaregiverLogger')
            logger.error(f"Speech error: {str(e)}")
            engine = None

recognizer = sr.Recognizer()

# Initialize MediaPipe Object Detection
model_path = 'efficientdet_lite0.tflite'
if not os.path.exists(model_path):
    logger = logging.getLogger('CaregiverLogger')
    logger.error(f"Model file {model_path} not found")
    raise FileNotFoundError(f"Model file {model_path} not found")
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.ObjectDetectorOptions(
    base_options=base_options,
    score_threshold=0.5,
    category_allowlist=[
        "person", "car", "truck", "motorcycle", "bicycle",
        "tree", "bench", "traffic light", "stop sign",
        "pole", "debris", "barrier", "table", "chair",
        "sofa", "bed"
    ]
)
object_detector = vision.ObjectDetector.create_from_options(options)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize Sound Recognition (YAMNet)
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# Parameters
KNOWN_WIDTHS = {
    "car": 1.8, "truck": 2.5, "motorcycle": 0.8, "bicycle": 0.7,
    "person": 0.5, "tree": 1.0, "stop sign": 0.75, "bench": 1.5,
    "traffic light": 0.5, "pole": 0.3, "debris": 0.5, "barrier": 1.0,
    "pothole": 0.5, "curb": 0.3, "stairs": 1.0, "table": 1.2,
    "chair": 0.6, "sofa": 2.0, "bed": 1.9
}
FOCAL_LENGTH = 1000
FRAME_WIDTH = 320
FRAME_HEIGHT = 180

# Indoor navigation map (example graph)
indoor_map = nx.Graph()
indoor_map.add_nodes_from([
    ("living_room", {"features": None}),
    ("kitchen", {"features": None}),
    ("bedroom", {"features": None}),
    ("bathroom", {"features": None})
])
indoor_map.add_edges_from([
    ("living_room", "kitchen", {"distance": 5}),
    ("kitchen", "bathroom", {"distance": 3}),
    ("living_room", "bedroom", {"distance": 4})
])

# User and caregiver data
user_data = {
    "user_id": str(uuid.uuid4()),
    "caregiver_phone": "caregiver_phone_number",
    "destination": None,
    "location": None,
    "behavior": "No user detected",
    "indoor_location": "unknown",
    "find_object": None
}

# Behavior tracking
user_positions = deque(maxlen=10)
user_head_orientations = deque(maxlen=5)
user_speeds = deque(maxlen=10)
prev_person_positions = deque(maxlen=10)
last_announce_time = time.time()
emergency_detected = False

# Detection history
detection_history = {
    "objects": deque(maxlen=5),
    "obstacles": deque(maxlen=5),
    "crowd_density": deque(maxlen=5),
    "text": deque(maxlen=5),
    "moving_objects": deque(maxlen=5),
    "user_behavior": deque(maxlen=5),
    "person_near_objects": deque(maxlen=5),
    "colors": deque(maxlen=5),
    "currency": deque(maxlen=5),
    "signage": deque(maxlen=5),
    "light_level": deque(maxlen=5),
    "scene": deque(maxlen=5),
    "emergency": deque(maxlen=5),
    "sounds": deque(maxlen=5)
}

# Optimization variables
last_ocr_time = time.time()
ocr_interval = 10
last_optical_flow_time = time.time()
optical_flow_interval = 0.2
last_location_update = time.time()
location_update_interval = 5
last_sound_time = time.time()
sound_interval = 10
prev_gray = None
alerts = []
navigation_steps = []
prev_frame = None
log_update_thread = None
log_update_running = True
last_log_update = time.time()
log_update_interval = 0.9

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
                logger = logging.getLogger('CaregiverLogger')
                logger.info(f"Downloaded {name}.png")
            except Exception as e:
                logger = logging.getLogger('CaregiverLogger')
                logger.error(f"Failed to download {name}.png: {str(e)}")

download_3d_objects()
arrow_texture = cv2.imread("3d_objects/arrow.png", cv2.IMREAD_UNCHANGED)
cone_texture = cv2.imread("3d_objects/cone.png", cv2.IMREAD_UNCHANGED)
marker_texture = cv2.imread("3d_objects/marker.png", cv2.IMREAD_UNCHANGED)

# Helper functions for accessibility features
def detect_color(frame):
    try:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        hue = np.argmax(hist)
        hsv_color = np.uint8([[[hue, 255, 255]]])
        rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)[0][0]
        rgb_color = rgb_color / 255.0
        h, s, v = colorsys.rgb_to_hsv(rgb_color[0], rgb_color[1], rgb_color[2])
        if v < 0.2:
            return "black"
        elif s < 0.2 and v > 0.8:
            return "white"
        elif h < 10 or h > 350:
            return "red"
        elif 10 <= h < 30:
            return "orange"
        elif 30 <= h < 90:
            return "yellow"
        elif 90 <= h < 150:
            return "green"
        elif 150 <= h < 270:
            return "blue"
        else:
            return "purple"
    except Exception as e:
        logger = logging.getLogger(f'CaregiverLogger_{session.get("caregiver_username", "unknown")}_{session["username"]}')
        logger.error(f"Color detection error: {str(e)}")
        return None

def detect_currency(frame):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return None  # Placeholder
    except Exception as e:
        logger = logging.getLogger(f'CaregiverLogger_{session.get("caregiver_username", "unknown")}_{session["username"]}')
        logger.error(f"Currency detection error: {str(e)}")
        return None

def recognize_place(frame):
    try:
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(frame, None)
        if descriptors is None:
            return "unknown"
        for place in indoor_map.nodes:
            if indoor_map.nodes[place]["features"] is None:
                indoor_map.nodes[place]["features"] = descriptors
            dist = cdist(descriptors, indoor_map.nodes[place]["features"], metric='hamming')
            if np.mean(dist) < 0.1:
                return place
        return "unknown"
    except Exception as e:
        logger = logging.getLogger(f'CaregiverLogger_{session.get("caregiver_username", "unknown")}_{session["username"]}')
        logger.error(f"Place recognition error: {str(e)}")
        return "unknown"

def navigate_indoors(start, destination):
    try:
        path = nx.astar_path(indoor_map, start, destination, weight='distance')
        steps = []
        for i in range(len(path) - 1):
            dist = indoor_map[path[i]][path[i+1]]['distance']
            steps.append(f"Move to {path[i+1]} in {dist} meters")
        return steps
    except Exception as e:
        logger = logging.getLogger(f'CaregiverLogger_{session.get("caregiver_username", "unknown")}_{session["username"]}')
        logger.error(f"Indoor navigation error: {str(e)}")
        return []

def detect_light_level(gray):
    try:
        intensity = np.mean(gray)
        if intensity > 150:
            return "bright"
        elif intensity > 50:
            return "dim"
        else:
            return "dark"
    except Exception as e:
        logger = logging.getLogger(f'CaregiverLogger_{session.get("caregiver_username", "unknown")}_{session["username"]}')
        logger.error(f"Light level detection error: {str(e)}")
        return None

def locate_object(detections, target):
    try:
        for detection in detections.detections:
            if detection.categories[0].category_name == target:
                distance = estimate_distance(detection.bounding_box.width, target)
                direction = get_direction(detection.bounding_box, FRAME_WIDTH, FRAME_HEIGHT)
                return distance, direction
        return None, None
    except Exception as e:
        logger = logging.getLogger(f'CaregiverLogger_{session.get("caregiver_username", "unknown")}_{session["username"]}')
        logger.error(f"Object locator error: {str(e)}")
        return None, None

def describe_scene(detections, place, crowd_density):
    try:
        objects = [d.categories[0].category_name for d in detections.detections]
        person_count = crowd_density[2]
        desc = f"You are in a {place} with {', '.join(objects) if objects else 'no objects'} and {person_count} people."
        return desc
    except Exception as e:
        logger = logging.getLogger(f'CaregiverLogger_{session.get("caregiver_username", "unknown")}_{session["username"]}')
        logger.error(f"Scene description error: {str(e)}")
        return None

def detect_emergency(pose_landmarks, detections, crowd_density):
    try:
        if pose_landmarks:
            left_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
            if left_hip.y > 0.8 and right_hip.y > 0.8:
                return "Fall detected"
        for detection in detections.detections:
            distance = estimate_distance(detection.bounding_box.width, detection.categories[0].category_name)
            if distance and distance < 1:
                return f"Close object: {detection.categories[0].category_name}"
        if crowd_density[0] == "High":
            return "Crowded area"
        return None
    except Exception as e:
        logger = logging.getLogger(f'CaregiverLogger_{session.get("caregiver_username", "unknown")}_{session["username"]}')
        logger.error(f"Emergency detection error: {str(e)}")
        return None

def navigate_crowd(detections):
    try:
        person_detections = [d for d in detections.detections if d.categories[0].category_name == "person"]
        if not person_detections:
            return None
        centers = [(d.bounding_box.origin_x + d.bounding_box.width // 2) / FRAME_WIDTH for d in person_detections]
        if np.mean(centers) < 0.4:
            return "right"
        elif np.mean(centers) > 0.6:
            return "left"
        return "forward"
    except Exception as e:
        logger = logging.getLogger(f'CaregiverLogger_{session.get("caregiver_username", "unknown")}_{session["username"]}')
        logger.error(f"Crowd navigation error: {str(e)}")
        return None

def detect_sound():
    try:
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
        frames = []
        for _ in range(int(16000 / 1024 * 1)):
            data = stream.read(1024, exception_on_overflow=False)
            frames.append(np.frombuffer(data, dtype=np.int16))
        stream.stop_stream()
        stream.close()
        p.terminate()
        audio = np.concatenate(frames).astype(np.float32) / 32768.0
        scores, _, _ = yamnet_model(audio)
        sound_id = np.argmax(scores[0])
        sound_labels = ["doorbell", "car horn", "footsteps"]
        if sound_id < len(sound_labels):
            return sound_labels[sound_id]
        return None
    except Exception as e:
        logger = logging.getLogger(f'CaregiverLogger_{session.get("caregiver_username", "unknown")}_{session["username"]}')
        logger.error(f"Sound detection error: {str(e)}")
        return None

def get_direction(bbox, frame_width, frame_height):
    try:
        center_x = bbox.origin_x + bbox.width // 2
        center_y = bbox.origin_y + bbox.height // 2
        x_ratio = center_x / frame_width
        y_ratio = center_y / frame_height
        if y_ratio < 0.4:
            if x_ratio < 0.3:
                return "front-left"
            elif x_ratio > 0.7:
                return "front-right"
            else:
                return "front"
        elif y_ratio > 0.6:
            if x_ratio < 0.3:
                return "back-left"
            elif x_ratio > 0.7:
                return "back-right"
            else:
                return "back"
        else:
            if x_ratio < 0.3:
                return "left"
            elif x_ratio > 0.7:
                return "right"
            else:
                return "center"
    except Exception as e:
        logger = logging.getLogger(f'CaregiverLogger_{session.get("caregiver_username", "unknown")}_{session["username"]}')
        logger.error(f"Direction calculation error: {str(e)}")
        return "unknown"

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.json.get('username').lower()
        email = request.json.get('email').lower()
        password = request.json.get('password').lower()
        if not username or not email or not password:
            return jsonify({'message': 'All fields are required'}), 400
        try:
            conn = get_db_connection()
            c = conn.cursor()
            hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
            c.execute('INSERT INTO users (username, email, password, role) VALUES (?, ?, ?, ?)',
                      (username, email, hashed_password, 'caregiver'))
            conn.commit()
            conn.close()
            return jsonify({'success': True, 'message': 'Signup successful'}), 201
        except sqlite3.IntegrityError:
            conn.close()
            return jsonify({'message': 'Username or email already exists'}), 400
        except Exception as e:
            conn.close()
            return jsonify({'message': f'An error occurred: {str(e)}'}), 500
    return render_template('caregiver_signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.json.get('username').lower()
        password = request.json.get('password').lower()
        if not username or not password:
            return jsonify({'message': 'Username and password are required'}), 400
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('SELECT id, password, role FROM users WHERE username = ?', (username,))
        user = c.fetchone()
        conn.close()
        if user and check_password_hash(user['password'], password):
            session['username'] = username
            session['user_id'] = user['id']
            session['role'] = user['role']
            return jsonify({'success': True, 'message': 'Login successful', 'redirect': url_for('caretakers')}), 200
        else:
            return jsonify({'message': 'Invalid username or password'}), 401
    return render_template('caregiver_login.html')

@app.route('/caretakers')
def caretakers():
    if 'username' not in session or session.get('role') != 'caregiver':
        flash('Please log in as a caregiver to access this page', 'error')
        return redirect(url_for('login'))
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT username FROM users_info WHERE caregiver_username = ?', (session['username'],))
    users = [{'username': row['username']} for row in c.fetchall()]
    conn.close()
    return render_template('caretaker.html', users=users, current_user=session['username'])

@app.route('/create-caretaker', methods=['GET', 'POST'])
def create_caretaker():
    if 'username' not in session or session.get('role') != 'caregiver':
        return jsonify({'message': 'Please log in as a caregiver to create a caretaker'}), 401
    if request.method == 'POST':
        username = request.json.get('username').lower()
        speech_credential = request.json.get('speech_credential').lower()
        if not username or not speech_credential:
            return jsonify({'message': 'Username and speech credential are required'}), 400
        conn = get_db_connection()
        c = conn.cursor()
        try:
            c.execute('SELECT username FROM users_info WHERE username = ?', (username,))
            if c.fetchone():
                conn.close()
                return jsonify({'message': 'Username already exists'}), 400
            c.execute('INSERT INTO users_info (username, speech_credential, caregiver_username) VALUES (?, ?, ?)',
                      (username, speech_credential, session['username']))
            conn.commit()
            conn.close()
            configure_logging(session['username'], username)
            return jsonify({'success': True, 'message': 'Caretaker created successfully'}), 201
        except sqlite3.Error as e:
            conn.close()
            return jsonify({'message': f'Database error: {str(e)}'}), 500
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT username FROM users_info WHERE caregiver_username = ?', (session['username'],))
    caretakers = [row['username'] for row in c.fetchall()]
    conn.close()
    return render_template('user_signup.html', caretakers=caretakers, current_user=session['username'])

@app.route('/caretaker_login')
def care_login():
    return render_template('user_login.html')

@app.route('/speech-login', methods=['POST'])
def speech_login():
    username = request.json.get('speech_credential').lower()
    logger = logging.getLogger('CaregiverLogger')
    if not username:
        logger.error("No speech credential provided in speech login")
        return jsonify({'message': 'Speech credential is required'}), 400
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute('SELECT username, caregiver_username FROM users_info WHERE username = ?', (username,))
        user = c.fetchone()
        if user:
            configure_logging(user['caregiver_username'], user['username'])
            logger = logging.getLogger(f'CaregiverLogger_{user["caregiver_username"]}_{user["username"]}')
            session['username'] = user['username']
            session['caregiver_username'] = user['caregiver_username']
            session['role'] = 'caretaker'
            logger.info(f"Caretaker {user['username']} logged in successfully")
            conn.close()
            return jsonify({'success': True, 'message': 'Login successful', 'redirect': url_for('vision')}), 200
        else:
            logger.error(f"Invalid speech credential: {username}")
            conn.close()
            return jsonify({'message': 'Invalid speech credential'}), 401
    except Exception as e:
        logger.error(f"Error during speech login: {str(e)}")
        conn.close()
        return jsonify({'message': 'An error occurred during login'}), 500

def login_required(f):
    def wrap(*args, **kwargs):
        if not session.get('username'):
            return redirect(url_for('care_login'))
        return f(*args, **kwargs)
    wrap.__name__ = f.__name__
    return wrap

@app.route('/vision')
@login_required
def vision():
    if session.get('role') != 'caretaker':
        flash('Please log in as a caretaker to access this page', 'error')
        return redirect(url_for('care_login'))
    logger = logging.getLogger(f'CaregiverLogger_{session.get("caregiver_username", "unknown")}_{session["username"]}')
    logger.info(f"Caretaker {session['username']} accessed vision page")
    return render_template('ar_.html')

log_date_now = datetime.now()
@app.route('/caregiver_dashboard/<caretaker_username>')
@login_required
def caregiver_dashboard(caretaker_username):
    global log_update_thread, log_update_running
    if session.get('role') != 'caregiver':
        flash('Please log in as a caregiver to access this page', 'error')
        return redirect(url_for('login'))
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT username, caregiver_username FROM users_info WHERE username = ?', (caretaker_username,))
    caretaker = c.fetchone()
    conn.close()
    if not caretaker or caretaker['caregiver_username'] != session['username']:
        flash('Unauthorized access to this caretaker', 'error')
        return redirect(url_for('caretakers'))
    logger = logging.getLogger(f'CaregiverLogger_{session["username"]}_{caretaker_username}')
    logger.info(f"Caregiver {session['username']} accessed dashboard for caretaker {caretaker_username}")
    log_files = glob.glob(f"logs/caregiver_{session['username']}_caretaker_{caretaker_username}.log")
    processed_logs = []
    log_file = None
    if log_files:
        log_file = log_files[0]
        try:
            with open(log_file, 'r') as f:
                logs = [line.strip() for line in f.readlines() if line.strip()][-20:]
            for log in logs:
                try:
                    parts = log.split(' - ', 2)
                    if len(parts) != 3:
                        continue
                    timestamp, log_type, message = parts
                    log_type = log_type.strip()
                    log_class = 'secondary'
                    if 'emergency' in message.lower() or 'warning' in log_type.lower():
                        log_class = 'danger'
                    elif 'error' in log_type.lower():
                        log_class = 'warning'
                    elif 'info' in log_type.lower():
                        log_class = 'info'
                    metrics = {}
                    if "Crowd density" in message:
                        metrics['crowd_density'] = message.split("Crowd density: ")[1].split(",")[0]
                    if "Person count" in message:
                        metrics['person_count'] = message.split("Person count: ")[1].split(",")[0]
                    if "Avg distance" in message:
                        metrics['avg_distance'] = message.split("Avg distance: ")[1].split("m")[0]
                    if "User behavior" in message:
                        metrics['behavior'] = message.split("User behavior: ")[1].split(",")[0]
                    if "Speed" in message:
                        metrics['speed'] = message.split("Speed: ")[1]
                    processed_logs.append({
                        'timestamp': timestamp.strip(),
                        'type': log_type,
                        'message': message.strip(),
                        'class': log_class,
                        'metrics': metrics
                    })
                except Exception as e:
                    logger.error(f"Error parsing log line: {log} - {str(e)}")
                    continue
        except Exception as e:
            logger.error(f"Error reading log file {log_file}: {str(e)}")

    if log_update_thread is None:
        log_update_running = True
        log_update_thread = threading.Thread(target=lambda: log_updater(caretaker_username), daemon=True)
        log_update_thread.start()
    return render_template(
        'caregiver_dashboard.html',
        username=caretaker_username,
        logs=processed_logs,
        log_file=log_file,
        last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        log_date_now=datetime.now()
    )

@app.route('/logout')
def logout():
    username = session.get('username')
    caregiver_username = session.get('caregiver_username')
    logger = logging.getLogger(f'CaregiverLogger_{caregiver_username}_{username}' if caregiver_username else 'CaregiverLogger')
    session.pop('username', None)
    session.pop('user_id', None)
    session.pop('caregiver_username', None)
    session.pop('role', None)
    speak("Goodbye. Logging out.")
    logger.info("User logged out")
    socketio.emit('logout', namespace=f'/video_feed')
    return redirect('/')

@app.route('/get_logs/<caretaker_username>')
@login_required
def get_logs(caretaker_username):
    if session.get('role') != 'caregiver':
        return jsonify({"error": "Unauthorized access, must be a caregiver"}), 403
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('SELECT username, caregiver_username FROM users_info WHERE username = ?', (caretaker_username,))
        caretaker = c.fetchone()
        conn.close()
        if not caretaker or caretaker['caregiver_username'] != session['username']:
            return jsonify({"error": "Unauthorized access to this caretaker's logs"}), 403
        logger = logging.getLogger(f'CaregiverLogger_{session["username"]}_{caretaker_username}')
        log_files = glob.glob(f"logs/caregiver_{session['username']}_caretaker_{caretaker_username}.log")
        logger.info(f"Found log files: {log_files}")
        if not log_files:
            logger.warning(f"No log files found for caretaker {caretaker_username}")
            return jsonify({"error": "No log files found for this caretaker"})
        latest_log = log_files[0]
        logger.info(f"Reading latest log file: {latest_log}")
        with open(latest_log, 'r') as f:
            logs = [line.strip() for line in f.readlines() if line.strip()]
        processed_logs = []
        for log in logs:
            try:
                parts = log.split(' - ', 2)
                if len(parts) != 3:
                    continue
                timestamp, log_type, message = parts
                log_type = log_type.strip()
                log_class = 'secondary'
                if 'emergency' in message.lower() or 'warning' in log_type.lower():
                    log_class = 'danger'
                elif 'error' in log_type.lower():
                    log_class = 'warning'
                elif 'info' in log_type.lower():
                    log_class = 'info'
                metrics = {}
                if "Crowd density" in message:
                    metrics['crowd_density'] = message.split("Crowd density: ")[1].split(",")[0]
                if "Person count" in message:
                    metrics['person_count'] = message.split("Person count: ")[1].split(",")[0]
                if "Avg distance" in message:
                    metrics['avg_distance'] = message.split("Avg distance: ")[1].split("m")[0]
                if "User behavior" in message:
                    metrics['behavior'] = message.split("User behavior: ")[1].split(",")[0]
                if "Speed" in message:
                    metrics['speed'] = message.split("Speed: ")[1]
                processed_logs.append({
                    'timestamp': timestamp.strip(),
                    'type': log_type,
                    'message': message.strip(),
                    'class': log_class,
                    'metrics': metrics
                })
            except Exception as e:
                logger.error(f"Error parsing log line: {log} - {str(e)}")
                continue
        return jsonify({
            'logs': processed_logs,
            'log_file': latest_log,
            'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    except Exception as e:
        logger.error(f"Get logs error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Helper functions
def listen_for_command():
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        speak("Listening...", priority=True)
        try:
            audio = recognizer.listen(source, timeout=5)
            command = recognizer.recognize_google(audio).lower()
            logger = logging.getLogger(f'CaregiverLogger_{session.get("caregiver_username", "unknown")}_{session["username"]}')
            logger.info(f"Voice command: {command}")
            return command
        except (sr.WaitTimeoutError, sr.UnknownValueError, sr.RequestError) as e:
            logger = logging.getLogger(f'CaregiverLogger_{session.get("caregiver_username", "unknown")}_{session["username"]}')
            logger.info(f"No voice command recognized: {str(e)}")
            return None

def estimate_distance(object_width_pixels, object_type):
    try:
        if object_type not in KNOWN_WIDTHS:
            return None
        real_width = KNOWN_WIDTHS[object_type]
        distance = (real_width * FOCAL_LENGTH) / object_width_pixels if object_width_pixels > 0 else None
        return round(distance, 1) if distance else None
    except Exception as e:
        logger = logging.getLogger(f'CaregiverLogger_{session.get("caregiver_username", "unknown")}_{session["username"]}')
        logger.error(f"Distance estimation error: {str(e)}")
        return None

def estimate_depth(landmarks, frame_width):
    try:
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
    except Exception as e:
        logger = logging.getLogger(f'CaregiverLogger_{session.get("caregiver_username", "unknown")}_{session["username"]}')
        logger.error(f"Depth estimation error: {str(e)}")
        return None

def classify_vehicle(label):
    if label == "car":
        return "4-wheeler"
    elif label == "truck":
        return "6-wheeler"
    elif label in ["motorcycle", "bicycle"]:
        return "2-wheeler"
    return label

def analyze_user_behavior(pose_landmarks, frame_width, frame_height, prev_positions, prev_orientations, speeds):
    try:
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
        logger = logging.getLogger(f'CaregiverLogger_{session.get("caregiver_username", "unknown")}_{session["username"]}')
        logger.info(f"User behavior: {behavior}, Speed: {avg_speed:.2f}")
        return behavior, avg_speed, gesture
    except Exception as e:
        logger = logging.getLogger(f'CaregiverLogger_{session.get("caregiver_username", "unknown")}_{session["username"]}')
        logger.error(f"User behavior analysis error: {str(e)}")
        return "No user detected", 0, None

def predict_person_behavior(pose_landmarks, prev_positions, frame_width):
    try:
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
    except Exception as e:
        logger = logging.getLogger(f'CaregiverLogger_{session.get("caregiver_username", "unknown")}_{session["username"]}')
        logger.error(f"Person behavior prediction error: {str(e)}")
        return "No person detected", None

def detect_speed_breakers(frame, gray):
    try:
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y1 - y2) < 10 and abs(x1 - x2) > 50:
                    return True
        return False
    except Exception as e:
        logger = logging.getLogger(f'CaregiverLogger_{session.get("caregiver_username", "unknown")}_{session["username"]}')
        logger.error(f"Speed breaker detection error: {str(e)}")
        return False

def detect_potholes(frame, gray):
    try:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 500:
                return True
        return False
    except Exception as e:
        logger = logging.getLogger(f'CaregiverLogger_{session.get("caregiver_username", "unknown")}_{session["username"]}')
        logger.error(f"Pothole detection error: {str(e)}")
        return False

def detect_curbs(frame, gray):
    try:
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=25, maxLineGap=5)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y1 - y2) > 25 and abs(x1 - x2) < 10:
                    return True
        return False
    except Exception as e:
        logger = logging.getLogger(f'CaregiverLogger_{session.get("caregiver_username", "unknown")}_{session["username"]}')
        logger.error(f"Curb detection error: {str(e)}")
        return False

def detect_stairs(frame, gray):
    try:
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=40, maxLineGap=10)
        if lines is not None:
            horizontal_lines = sum(1 for line in lines if abs(line[0][1] - line[0][3]) < 10)
            if horizontal_lines > 3:
                return True
        return False
    except Exception as e:
        logger = logging.getLogger(f'CaregiverLogger_{session.get("caregiver_username", "unknown")}_{session["username"]}')
        logger.error(f"Stairs detection error: {str(e)}")
        return False

def is_road_context(frame, speed_breaker, gray):
    try:
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
        line_count = len(lines) if lines is not None else 0
        return line_count > 5 or speed_breaker or line_count < 10
    except Exception as e:
        logger = logging.getLogger(f'CaregiverLogger_{session.get("caregiver_username", "unknown")}_{session["username"]}')
        logger.error(f"Road context detection error: {str(e)}")
        return False

def read_text(frame, gray):
    try:
        text = pytesseract.image_to_string(gray, config='--psm 6')
        text = text.strip() if text.strip() else None
        if text and re.match(r'^[a-zA-Z0-9\s.,!?]+$', text):
            logger = logging.getLogger(f'CaregiverLogger_{session.get("caregiver_username", "unknown")}_{session["username"]}')
            logger.info(f"OCR text: {text}")
            return text
        return None
    except Exception as e:
        logger = logging.getLogger(f'CaregiverLogger_{session.get("caregiver_username", "unknown")}_{session["username"]}')
        logger.error(f"OCR error: {str(e)}")
        return None

def get_directions(destination):
    try:
        g = geocoder.osm(destination)
        if g.ok:
            user_loc = geocoder.ip('me').latlng
            dest_loc = g.latlng
            distance = math.sqrt((dest_loc[0] - user_loc[0]) ** 2 + (dest_loc[1] - user_loc[1]) ** 2) * 111
            steps = ["Turn left in 50 meters", "Continue straight for 200 meters"]
            logger = logging.getLogger(f'CaregiverLogger_{session.get("caregiver_username", "unknown")}_{session["username"]}')
            logger.info(f"Navigation steps for {destination}: {steps}")
            return {"distance": round(distance, 1), "steps": steps}
        return None
    except Exception as e:
        logger = logging.getLogger(f'CaregiverLogger_{session.get("caregiver_username", "unknown")}_{session["username"]}')
        logger.error(f"Geocoder error: {str(e)}")
        return None

def analyze_crowd_density(detections):
    try:
        person_count = sum(1 for detection in detections.detections if detection.categories[0].category_name == "person")
        avg_distance = 0
        if person_count > 0:
            distances = [
                estimate_distance(detection.bounding_box.width, "person")
                for detection in detections.detections
                if detection.categories[0].category_name == "person" and estimate_distance(detection.bounding_box.width, "person") is not None
            ]
            avg_distance = np.mean(distances) if distances else 0
        if person_count > 10:
            density = "High"
            warning = f"Crowded area with {person_count} people detected."
            speak(warning)
        elif person_count >= 5 or (person_count > 0 and avg_distance < 3):
            density = "High"
            warning = "Crowded area ahead, slow down"
        elif person_count >= 3 or (person_count > 0 and avg_distance < 5):
            density = "Medium"
            warning = "Moderately crowded area, proceed with caution"
        else:
            density = "Low"
            warning = None
        logger = logging.getLogger(f'CaregiverLogger_{session.get("caregiver_username", "unknown")}_{session["username"]}')
        logger.info(f"Crowd density: {density}, Person count: {person_count}, Avg distance: {avg_distance:.1f}m")
        return density, warning, person_count, avg_distance
    except Exception as e:
        logger = logging.getLogger(f'CaregiverLogger_{session.get("caregiver_username", "unknown")}_{session["username"]}')
        logger.error(f"Crowd density analysis error: {str(e)}")
        return "Low", None, 0, 0

def detect_person_near_object(detections, pose_landmarks, frame_width, frame_height):
    try:
        person_near_objects = []
        person_detections = [d for d in detections.detections if d.categories[0].category_name == "person"]
        object_detections = [d for d in detections.detections if d.categories[0].category_name in ["table", "chair", "sofa", "bed"]]
        for person in person_detections:
            person_bbox = person.bounding_box
            person_center = (person_bbox.origin_x + person_bbox.width // 2, person_bbox.origin_y + person_bbox.height // 2)
            person_distance = estimate_distance(person_bbox.width, "person")
            for obj in object_detections:
                obj_bbox = obj.bounding_box
                obj_center = (obj_bbox.origin_x + obj_bbox.width // 2, obj_bbox.origin_y + obj_bbox.height // 2)
                obj_distance = estimate_distance(obj_bbox.width, obj.categories[0].category_name)
                pixel_distance = math.sqrt((person_center[0] - obj_center[0]) ** 2 + (person_center[1] - obj_center[1]) ** 2)
                if pixel_distance < 50 and person_distance and obj_distance and abs(person_distance - obj_distance) < 1.0:
                    interaction = "near"
                    if pose_landmarks:
                        left_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
                        right_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
                        hip_height = (left_hip.y + right_hip.y) / 2 * frame_height
                        if obj.categories[0].category_name in ["chair", "sofa"] and hip_height > frame_height * 0.6:
                            interaction = "sitting on"
                        elif obj.categories[0].category_name == "bed" and hip_height > frame_height * 0.7:
                            interaction = "lying on"
                    person_near_objects.append({
                        "person_distance": person_distance,
                        "object_type": obj.categories[0].category_name,
                        "object_distance": obj_distance,
                        "interaction": interaction
                    })
        return person_near_objects
    except Exception as e:
        logger = logging.getLogger(f'CaregiverLogger_{session.get("caregiver_username", "unknown")}_{session["username"]}')
        logger.error(f"Person near object error: {str(e)}")
        return []

def compute_optical_flow(prev_frame, curr_frame):
    global prev_gray
    if prev_frame is None:
        return None, None
    try:
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        prev_gray = curr_gray
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        avg_mag = np.mean(mag)
        moving_objects = []
        if avg_mag > 1.0:
            h, w = curr_frame.shape[:2]
            for detection in object_detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=curr_frame)).detections:
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
        logger = logging.getLogger(f'CaregiverLogger_{session.get("caregiver_username", "unknown")}_{session["username"]}')
        logger.error(f"Optical flow error: {str(e)}")
        return None, None

def draw_object_detections(frame, detections, user_behavior, navigation_steps, road_context, crowd_density, moving_objects):
    try:
        annotated_image = frame.copy()
        height, width = frame.shape[:2]
        if road_context and navigation_steps:
            annotated_image = overlay_3d_object(annotated_image, arrow_texture, (25, 25), size=10)
            logger = logging.getLogger(f'CaregiverLogger_{session.get("caregiver_username", "unknown")}_{session["username"]}')
            logger.info("Applied AR arrow overlay")
        if road_context and detect_lanes(frame):
            annotated_image = overlay_3d_object(annotated_image, cone_texture, (width // 2, height - 50), size=10)
            logger = logging.getLogger(f'CaregiverLogger_{session.get("caregiver_username", "unknown")}_{session["username"]}')
            logger.info("Applied AR cone overlay")
        return annotated_image
    except Exception as e:
        logger = logging.getLogger(f'CaregiverLogger_{session.get("caregiver_username", "unknown")}_{session["username"]}')
        logger.error(f"Draw detections error: {str(e)}")
        return frame

def announce_detections(detections, user_behavior, person_behavior, text, obstacles, navigation_steps, crowd_density, moving_objects, person_near_objects, color, currency, signage, light_level, object_distance, object_direction, scene_desc, emergency, crowd_direction, sound):
    global last_announce_time, detection_history
    if time.time() - last_announce_time < 5:
        return
    try:
        announcements = []
        current_objects = []
        for detection in detections.detections:
            category = detection.categories[0]
            distance = estimate_distance(detection.bounding_box.width, category.category_name)
            if distance and distance < 5 and category.score > 0.7:
                direction = get_direction(detection.bounding_box, FRAME_WIDTH, FRAME_HEIGHT)
                obj_type = classify_vehicle(category.category_name)
                obj_desc = f"{obj_type} on your {direction} at {distance}m"
                current_objects.append(obj_desc)
                if obj_desc not in detection_history["objects"]:
                    announcements.append(f"{obj_type} detected on your {direction} at {distance} meters")
                    logger = logging.getLogger(f'CaregiverLogger_{session.get("caregiver_username", "unknown")}_{session["username"]}')
                    logger.info(f"Detection: {obj_type}, Direction: {direction}, Distance: {distance}m")
        detection_history["objects"].append(current_objects)
        current_obstacles = []
        for obstacle, detected, distance in obstacles:
            if detected and distance and distance < 5:
                obs_desc = f"{obstacle} on your front at {distance}m"
                current_obstacles.append(obs_desc)
                if obs_desc not in detection_history["obstacles"]:
                    announcements.append(f"{obstacle} detected on your front at {distance} meters")
        detection_history["obstacles"].append(current_obstacles)
        if user_behavior != "No user detected":
            if user_behavior not in detection_history["user_behavior"]:
                announcements.append(f"User behavior: {user_behavior}")
            detection_history["user_behavior"].append(user_behavior)
        if text:
            if text not in detection_history["text"]:
                announcements.append(f"Sign detected: {text}")
            detection_history["text"].append(text)
        if crowd_density[1]:
            crowd_desc = crowd_density[1]
            if crowd_desc not in detection_history["crowd_density"]:
                announcements.append(f"{crowd_desc}")
            detection_history["crowd_density"].append(crowd_desc)
        current_moving = []
        if moving_objects:
            for label, direction, _, _ in moving_objects:
                move_desc = f"{label} from {direction}"
                current_moving.append(move_desc)
                if move_desc not in detection_history["moving_objects"]:
                    announcements.append(f"{label} approaching from the {direction}")
        detection_history["moving_objects"].append(current_moving)
        current_person_near_objects = []
        for pno in person_near_objects:
            desc = f"Person {pno['interaction']} {pno['object_type']} at {pno['object_distance']} meters"
            current_person_near_objects.append(desc)
            if desc not in detection_history["person_near_objects"]:
                announcements.append(desc)
                logger.info(f"Person near object: {desc}")
        detection_history["person_near_objects"].append(current_person_near_objects)
        if color and color not in detection_history["colors"]:
            announcements.append(f"Dominant color: {color}")
            detection_history["colors"].append(color)
        if currency and currency not in detection_history["currency"]:
            announcements.append(f"Currency detected: {currency}")
            detection_history["currency"].append(currency)
        if light_level and light_level not in detection_history["light_level"]:
            announcements.append(f"Light level: {light_level}")
            detection_history["light_level"].append(light_level)
        if object_distance and object_direction:
            obj_loc_desc = f"{user_data['find_object']} located on your {object_direction} at {object_distance} meters"
            announcements.append(obj_loc_desc)
        if scene_desc and scene_desc not in detection_history["scene"]:
            announcements.append(scene_desc)
            detection_history["scene"].append(scene_desc)
        if emergency and emergency not in detection_history["emergency"]:
            announcements.append(f"Emergency: {emergency}")
            detection_history["emergency"].append(emergency)
            socketio.emit('emergency', {'message': emergency}, namespace=f'/caregiver_dashboard/{session["username"]}')
        if crowd_direction:
            announcements.append(f"Crowded area ahead, move {crowd_direction} to avoid")
        if sound and sound not in detection_history["sounds"]:
            announcements.append(f"Sound detected: {sound}")
            detection_history["sounds"].append(sound)
        if announcements:
            announcement = ". ".join(announcements[:5])
            speak(announcement, priority=emergency is not None)
            last_announce_time = time.time()
    except Exception as e:
        logger = logging.getLogger(f'CaregiverLogger_{session.get("caregiver_username", "unknown")}_{session["username"]}')
        logger.error(f"Announce detections error: {str(e)}")

def settings_menu():
    try:
        speak("Settings menu. Say a number from 1 to 4. 1: Navigate to a destination. 2: Update caregiver details. 3: Find an object. 4: Logout")
        command = listen_for_command()
        if not command:
            speak("No command recognized")
            return None
        if "1" in command or "one" in command:
            speak("Where do you wish to go?")
            destination = listen_for_command()
            if destination:
                user_data["destination"] = destination
                directions = get_directions(destination)
                if directions:
                    navigation_steps = directions["steps"]
                    speak(f"Destination set to {destination}. Distance is {directions['distance']} kilometers. Follow: {', '.join(directions['steps'])}")
                    logger = logging.getLogger(f'CaregiverLogger_{session.get("caregiver_username", "unknown")}_{session["username"]}').info(f"Navigation set to {destination}, Distance: {directions['distance']}km")
                    return navigation_steps
                speak("Could not find destination")
                return None
        elif "2" in command or "two" in command:
            speak("Say the caregiver phone number.")
            phone = listen_for_command()
            if phone:
                user_data["caregiver_phone"] = phone
                speak(f"Caregiver phone updated to {phone}")
                logger = logging.getLogger(f'CaregiverLogger_{session.get("caregiver_username", "unknown")}_{session["username"]}').info(f"Caregiver phone updated: to {phone}")
                return None
            return None
        elif "3" in command or "three" in command:
            speak("What object do you wish to find?")
            object = listen_for_command()
            if object:
                user_data["find_object"] = object
                speak(f"Looking for {object}")
                logger = logging.getLogger(f'CaregiverLogger_{session.get("caregiver_username", "unknown")}_{session["username"]}').info(f"Object search initiated: {object}")
                return None
            return None
        elif "4" in command or "four" in command:
            return "logout"
        else:
            speak("Invalid option")
            return None
    except Exception as e:
        logger = logging.getLogger(f'CaregiverLogger_{session.get("caregiver_username", "unknown")}_{session["username"]}').error(f"Settings menu error: {str(e)}")
        return None

def overlay_3d_object(frame, texture, position, size=10):
    if texture is None:
        return frame
    try:
        texture = cv2.resize(texture, (size, size))
        if texture.shape[2] == 4:
            alpha = texture[:, :, 3] / 255.0
            texture = texture[:, :, 0:3]
        else:
            alpha = np.ones((size, size)) * 0.7
        x, y = position
        y1, y2 = max(0, y - size // 2), min(frame.shape[0], y + size // 2)
        x1, x2 = max(0, x - size // 2), min(frame.shape[1], x + size // 2)
        if y2 - y1 > 0 and x2 - x1 > 0:
            texture_roi = cv2.resize(texture, (x2 - x1, y2 - y1))
            alpha_roi = cv2.resize(alpha, (x2 - x1, y2 - y1))
            for c in range(0, 3):
                frame[y1:y2, x1:x2, c] = (
                    alpha_roi * texture_roi[:, :, c] +
                    (1 - alpha_roi) * frame[y1:y2, x1:x2, c]
                )
        return frame
    except Exception as e:
        logger = logging.getLogger(f'CaregiverLogger_{session.get("caregiver_username", "unknown")}_{session["username"]}')
        logger.error(f"Overlay 3D object error: {str(e)}")
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
        logger = logging.getLogger(f'CaregiverLogger_{session.get("caregiver_username", "unknown")}_{session["username"]}')
        logger.error(f"Lane detection error: {str(e)}")
        return False

@socketio.on('connect', namespace='/video_feed')
def handle_connect():
    try:
        if not session.get('username') or session.get('role') != 'caretaker':
            emit('error', {'message': 'Unauthorized'}, namespace='/video_feed')
            return False
        logger = logging.getLogger(f'CaregiverLogger_{session.get("caregiver_username", "unknown")}_{session["username"]}')
        logger.info(f'WebSocket client connected')
        emit('connected', {'message': 'Connected'}, namespace='/video_feed')
    except Exception as e:
        logger = logging.getLogger(f'CaregiverLogger_{session.get("caregiver_username", "unknown")}_{session["username"]}')
        logger.error(f"WebSocket connect error: {str(e)}")

@socketio.on('disconnect', namespace='/video_feed')
def handle_disconnect():
    logger = logging.getLogger(f'CaregiverLogger_{session.get("caregiver_username", "unknown")}_{session["username"]}')
    logger.info(f'WebSocket client disconnected')

@socketio.on_error(namespace='/video_feed')
def handle_error(e):
    logger = logging.getLogger(f'CaregiverLogger_{session.get("caregiver_username", "unknown")}_{session["username"]}')
    logger.error(f"WebSocket error: {str(e)}")

@socketio.on('message', namespace='/video_feed')
def handle_frame(data):
    global last_ocr_time, last_optical_flow_time, last_location_update, prev_gray, alerts, navigation_steps, prev_frame, last_sound_time
    logger = logging.getLogger(f'CaregiverLogger_{session.get("caregiver_username", "unknown")}_{session["username"]}')
    logger.info("Received video frame from client")
    try:
        if not isinstance(data, str) or ',' not in data or not data.startswith('data:image'):
            logger.error("Invalid frame data format")
            raise ValueError("Invalid frame format")
        try:
            img_data = base64.b64decode(data.split(',')[1])
            npimg = np.frombuffer(img_data, dtype=np.uint8)
            frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Failed to decode image")
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        except Exception as e:
            logger.error(f"Frame decoding error: {str(e)}")
            emit('error', {'message': 'Failed to decode frame'}, namespace='/video_feed')
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if time.time() - last_location_update > location_update_interval:
            try:
                user_data["location"] = geocoder.ip('me').latlng
                user_data["indoor_location"] = recognize_place(frame)
                logger.info(f"Location updated: {user_data['location']}, indoor: {user_data['indoor_location']}")
                last_location_update = time.time()
            except Exception as e:
                logger.error(f"Location update error: {str(e)}")
        obstacle_results = [[] for _ in range(4)]
        def detect_obstacles(idx, func, name, obstacle_results):
            try:
                detected = func(frame, gray)
                distance = estimate_distance(
                    100 if name in ["Speed breaker", "Pothole"] else 50 if name == "Curb" else 200,
                    name.lower()) if detected else None
                obstacle_results[idx].append((name, detected, distance))
            except Exception as e:
                logger.error(f"Error detecting {name}: {str(e)}")
                obstacle_results[idx].append((name, False, None))
        obstacle_threads = [
            threading.Thread(target=detect_obstacles, args=(0, detect_speed_breakers, "Speed breaker", obstacle_results)),
            threading.Thread(target=detect_obstacles, args=(1, detect_potholes, "Pothole", obstacle_results)),
            threading.Thread(target=detect_obstacles, args=(2, detect_curbs, "Curb", obstacle_results)),
            threading.Thread(target=detect_obstacles, args=(3, detect_stairs, "Stairs", obstacle_results))
        ]
        for t in obstacle_threads:
            t.start()
        for t in obstacle_threads:
            t.join()
        obstacles = [result[0] for result in obstacle_results if result]
        try:
            color = detect_color(frame)
            currency = detect_currency(frame)
            signage = read_text(frame, gray)
            light_level = detect_light_level(gray)
        except Exception as e:
            logger.error(f"Feature detection error: {str(e)}")
            color = currency = signage = light_level = None
        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            detection_result = object_detector.detect(mp_image)
        except Exception as e:
            logger.error(f"Object detection error: {str(e)}")
            detection_result = type('obj', (), {'detections': []})()
        crowd_density = analyze_crowd_density(detection_result)
        crowd_direction = navigate_crowd(detection_result)
        moving_objects = None
        flow = None
        if time.time() - last_optical_flow_time > optical_flow_interval and prev_frame is not None:
            moving_objects, flow = compute_optical_flow(prev_frame, frame)
            last_optical_flow_time = time.time()
        sound = detect_sound() if time.time() - last_sound_time > sound_interval else None
        if sound:
            logger.info(f"Sound detected: {sound}")
            last_sound_time = time.time()
        try:
            pose_results = pose.process(frame)
        except Exception as e:
            logger.error(f"Pose estimation error: {str(e)}")
            pose_results = type('obj', (), {'pose_landmarks': None})()
        user_behavior = "No user detected"
        person_behavior = ("No person detected", None)
        if pose_results.pose_landmarks:
            user_behavior, user_speed, user_gesture = analyze_user_behavior(
                pose_results.pose_landmarks, FRAME_WIDTH, FRAME_HEIGHT, user_positions, user_head_orientations, user_speeds
            )
            person_behavior = predict_person_behavior(pose_results.pose_landmarks, prev_person_positions, FRAME_WIDTH)
            if person_behavior[1] is not None:
                prev_person_positions.append(person_behavior[1])
        person_near_objects = detect_person_near_object(detection_result, pose_results.pose_landmarks, FRAME_WIDTH, FRAME_HEIGHT)
        depth = estimate_depth(pose_results.pose_landmarks, FRAME_WIDTH)
        object_distance, object_direction = None, None
        if user_data["find_object"]:
            object_distance, object_direction = locate_object(detection_result, user_data["find_object"])
        scene_desc = describe_scene(detection_result, user_data["indoor_location"], crowd_density)
        emergency = detect_emergency(pose_results.pose_landmarks, detection_result, crowd_density)
        if user_data["destination"] and user_data["indoor_location"] != "unknown":
            navigation_steps = navigate_indoors(user_data["indoor_location"], user_data["destination"])
        text = None
        if time.time() - last_ocr_time > ocr_interval:
            text = read_text(frame, gray)
            last_ocr_time = time.time()
        emergency_detected = False
        for detection in detection_result.detections:
            category = detection.categories[0]
            distance = estimate_distance(detection.bounding_box.width, category.category_name)
            if distance and distance < 2 and category.score > 0.7:
                emergency_detected = True
                logger.info(f"Emergency detected: {category.category_name} too close at {distance}m")
                alerts.append(f"Warning: {category.category_name} detected at {distance} meters")
        speed_breaker_detected = any(o[1] for o in obstacles if o[0] == "Speed breaker")
        road_context = is_road_context(frame, speed_breaker_detected, gray)
        overlays = {"objects": [], "navigation": []}
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            category = detection.categories[0]
            label = classify_vehicle(category.category_name)
            distance = estimate_distance(bbox.width, category.category_name)
            if distance and distance < 8 and category.score > 0.7:
                direction = get_direction(bbox, FRAME_WIDTH, FRAME_HEIGHT)
                overlays["objects"].append({
                    "label": label,
                    "score": category.score,
                    "direction": direction,
                    "distance": round(distance, 1)
                })
        for obstacle, detected, distance in obstacles:
            if detected and distance and distance < 5:
                overlays["objects"].append({
                    "label": obstacle,
                    "score": 0.9,
                    "distance": round(distance, 1)
                })
        if navigation_steps and road_context:
            overlays["navigation"].append({"type": "arrow", "direction": "left", "x": 25, "y": 25})
        frame_with_detections = draw_object_detections(frame, detection_result, user_behavior, navigation_steps, road_context, crowd_density, moving_objects)
        alerts = []
        for obstacle, detected, distance in obstacles:
            if detected and distance and distance < 5:
                alerts.append(f"{obstacle} ahead at {distance} meters")
                for person_near_object in person_near_objects:  # Fixed variable name for clarity
                    alerts.append(
                        f"Person {person_near_object['interaction']} {person_near_object['object_type']} at {person_near_object['object_distance']} meters"
                    )
                if emergency:
                    alerts.append(f"Emergency: {emergency}")
                    emergency_detected = True

                try:  # Start of try block for frame encoding
                    _, buffer = cv2.imencode('.jpg', frame_with_detections)
                    frame_b64 = base64.b64encode(buffer).decode('utf-8')
                    frame_data = f"data:image/jpeg;base64,{frame_b64}"
                except Exception as e:  # Properly formatted except clause
                    logger.error(f"Frame encoding error: {str(e)}")
                    emit('error', {'message': 'Failed to encode frame'}, namespace='/video_feed')
                    return

                try:  # Separate try block for announce_detections
                    announce_detections(
                        detection_result,
                        user_behavior,
                        person_behavior,
                        text,
                        obstacles,
                        navigation_steps,
                        crowd_density,
                        moving_objects,
                        person_near_objects,
                        color,
                        currency,
                        signage,
                        light_level,
                        object_distance,
                        object_direction,
                        scene_desc,
                    )
                except Exception as e:  # Fixed: Removed extra parenthesis
                    logger.error(f"Announce detections error: {str(e)}")

                # Dashboard data preparation
                dashboard_data = {
                    'user_behavior': user_behavior,
                    'alerts': alerts[:5],
                    'objects': [f"{o['label']} at {o['distance']}m ({o['direction']})" for o in overlays["objects"]],
                    'emergency': emergency,
                    'crowd_density': crowd_density[0],
                    'person_count': crowd_density[2],
                    'avg_distance': round(crowd_density[3], 1) if crowd_density[3] else 0,
                    'scene': scene_desc,
                    'location': user_data["indoor_location"],
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

                socketio.emit(
                    'dashboard_update',
                    dashboard_data,
                    namespace=f'/caregiver_dashboard/{session["username"]}'
                )

                emit('frame', {
                    'image': frame_data,
                    'overlays': overlays,
                    'alerts': alerts,
                    'user_behavior': user_behavior,
                    'emergency': emergency
                }, namespace='/video_feed')

                prev_frame = frame.copy()

                if emergency_detected:
                    socketio.emit(
                        'emergency',
                        {'message': emergency or "Emergency situation detected"},
                        namespace=f'/caregiver_dashboard/{session["username"]}'
                    )
    except Exception as e:  # Outer try-except for entire frame processing
        logger.error(f"Frame processing error: {str(e)}")
        emit('error', {'message': str(e)}, namespace='/video_feed')

@socketio.on('voice_command', namespace='/video_feed')
def handle_voice_command(command):
    logger = logging.getLogger(f'CaregiverLogger_{session.get("caregiver_username", "unknown")}_{session["username"]}')
    logger.info(f"Received voice command: {command}")
    try:
        command = command.lower() if isinstance(command, str) else None
        if not command:
            speak("No command recognized")
            return
        if "settings" in command:
            result = settings_menu()
            if result == "logout":
                socketio.emit('logout', namespace='/video_feed')
                session.pop('username', None)
                session.pop('caregiver_username', None)
                session.pop('role', None)
                logger.info("User logged out via voice command")
            elif result:
                socketio.emit('navigation', {'steps': result}, namespace='/video_feed')
        elif "stop" in command:
            speak("Stopping current action")
            user_data["destination"] = None
            user_data["find_object"] = None
            navigation_steps.clear()
            logger.info("Stopped current action")
        elif "help" in command:
            speak("Available commands: settings, stop, help")
        else:
            speak("Command not recognized")
    except Exception as e:
        logger.error(f"Voice command error: {str(e)}")
        speak("Error processing command")


def log_updater(caretaker_username):
    global last_log_update
    logger = logging.getLogger(f'CaregiverLogger_{session.get("username", "unknown")}_{caretaker_username}')
    while log_update_running:
        try:
            if time.time() - last_log_update < log_update_interval:
                time.sleep(0.1)
                continue
            log_files = glob.glob(
                f"logs/caregiver_{session.get('username', 'unknown')}_caretaker_{caretaker_username}.log")
            if log_files:
                socketio.emit(
                    'log_update',
                    {'caretaker': caretaker_username},
                    namespace=f'/caregiver_dashboard/{caretaker_username}'
                )
            last_log_update = time.time()
        except Exception as e:
            logger.error(f"Log updater error: {str(e)}")
        time.sleep(0.5)


@app.route('/delete_caretaker/<caretaker_username>', methods=['POST'])
@login_required
def delete_caretaker(caretaker_username):
    if session.get('role') != 'caregiver':
        flash('Unauthorized access', 'error')
        return redirect(url_for('login'))
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT caregiver_username FROM users_info WHERE username = ?', (caretaker_username,))
    caretaker = c.fetchone()
    if not caretaker or caretaker['caregiver_username'] != session['username']:
        conn.close()
        flash('Unauthorized to delete this caretaker', 'error')
        return redirect(url_for('caretakers'))
    try:
        c.execute('DELETE FROM users_info WHERE username = ?', (caretaker_username,))
        conn.commit()
        log_file = f"logs/caregiver_{session['username']}_caretaker_{caretaker_username}.log"
        if os.path.exists(log_file):
            os.remove(log_file)
        logger = logging.getLogger(f'CaregiverLogger_{session["username"]}_{caretaker_username}')
        logger.info(f"Caretaker {caretaker_username} deleted by caregiver {session['username']}")
        flash('Caretaker deleted successfully', 'success')
    except Exception as e:
        conn.close()
        flash(f'Error deleting caretaker: {str(e)}', 'error')
    finally:
        conn.close()
    return redirect(url_for('caretakers'))


@app.route('/download_logs/<caretaker_username>')
@login_required
def download_logs(caretaker_username):
    if session.get('role') != 'caregiver':
        flash('Unauthorized access', 'error')
        return redirect(url_for('login'))
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT caregiver_username FROM users_info WHERE username = ?', (caretaker_username,))
    caretaker = c.fetchone()
    conn.close()
    if not caretaker or caretaker['caregiver_username'] != session['username']:
        flash("Unauthorized access to this caretaker�s logs", "error")
        return redirect(url_for('caretakers'))
    log_file = f"logs/caregiver_{session['username']}_caretaker_{caretaker_username}.log"
    logger = logging.getLogger(f'CaregiverLogger_{session["username"]}_{caretaker_username}')
    if os.path.exists(log_file):
        logger.info(f"Caregiver {session['username']} downloaded logs for caretaker {caretaker_username}")
        return send_file(log_file, as_attachment=True, download_name=f"caretaker_{caretaker_username}_logs.log")
    else:
        logger.warning(f"Log file not found for caretaker {caretaker_username}")
        flash('No logs found for this caretaker', 'error')
        return redirect(url_for('caretakers'))


@socketio.on('connect', namespace='/caregiver_dashboard')
def handle_dashboard_connect():
    if not session.get('username') or session.get('role') != 'caregiver':
        emit('error', {'message': 'Unauthorized'}, namespace='/caregiver_dashboard')
        return False
    logger = logging.getLogger(f'CaregiverLogger_{session.get("username", "unknown")}_general')
    logger.info('Caregiver dashboard WebSocket connected')


@socketio.on('disconnect', namespace='/caregiver_dashboard')
def handle_dashboard_disconnect():
    logger = logging.getLogger(f'CaregiverLogger_{session.get("username", "unknown")}_general')
    logger.info('Caregiver dashboard WebSocket disconnected')


if __name__ == '__main__':
    logger = logging.getLogger('CaregiverLogger')
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(stream_handler)
    local_ip = get_local_ip()
    logger.info(f"Server starting on http://{local_ip}:5000")
    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
    except Exception as e:
        logger.error(f"Server startup error: {str(e)}")