from flask import Flask, request, jsonify, session
from werkzeug.security import generate_password_hash, check_password_hash
from flask_socketio import SocketIO, emit
import sqlite3
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from datetime import datetime
import threading
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'  # Replace with a secure key
socketio = SocketIO(app)

# Database setup
DB_PATH = 'users.db'

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    speech_credential TEXT NOT NULL,
                    last_active TEXT DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'Active'
                )
            ''')
        conn.commit()


@app.route('/speech-signup', methods=['POST'])
def speech_signup():
    data = request.get_json()
    username = data.get('username')
    speech_credential = data.get('speech_credential')
    if not username or not speech_credential:
        return jsonify({'success': False, 'message': 'All fields are required'}), 400
    hashed_credential = generate_password_hash(speech_credential)
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
            if cursor.fetchone():
                return jsonify({'success': False, 'message': 'Username already exists'}), 400

            # Insert new user
            cursor.execute('''
                    INSERT INTO users (username, speech_credential, last_active, status)
                    VALUES (?, ?, ?, ?)
                ''', (username, hashed_credential, datetime.utcnow().isoformat(), 'Active'))
            conn.commit()
            return jsonify({'success': True, 'message': 'User registered successfully'}), 200
    except sqlite3.Error as e:
        return jsonify({'success': False, 'message': 'Registration failed'}), 500


@app.route('/speech-login', methods=['POST'])
def speech_login():
    data = request.get_json()
    speech_credential = data.get('speech_credential')

    if not speech_credential:
        return jsonify({'success': False, 'message': 'Speech credential is required'}), 400

    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, speech_credential FROM users')
            users = cursor.fetchall()

            for user in users:
                user_id, stored_credential = user
                if check_password_hash(stored_credential, speech_credential):
                    session['user_id'] = user_id
                    cursor.execute('''
                            UPDATE users SET last_active = ? WHERE id = ?
                        ''', (datetime.utcnow().isoformat(), user_id))
                    conn.commit()
                    return jsonify({'success': True, 'message': 'Login successful'}), 200

            return jsonify({'success': False, 'message': 'Invalid speech credential'}), 401
    except sqlite3.Error as e:
        return jsonify({'success': False, 'message': 'Login failed'}), 500


@app.route('/logout', methods=['POST'])
def logout():
    session.pop('user_id', None)
    return jsonify({'success': True, 'message': 'Logged out successfully'}), 200


@app.route('/api/user-data', methods=['GET'])
def get_user_data():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401

    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                    SELECT username, last_active, status FROM users WHERE id = ?
                ''', (user_id,))
            user = cursor.fetchone()
            if not user:
                return jsonify({'success': False, 'message': 'User not found'}), 404

            return jsonify({
                'success': True,
                'user': {
                    'username': user[0],
                    'last_active': user[1],
                    'status': user[2]
                }
            }), 200
    except sqlite3.Error as e:
        return jsonify({'success': False, 'message': 'Error fetching user data'}), 500


# Obstacle Detection Setup
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

KNOWN_WIDTHS = {
    "car": 1.8,
    "truck": 2.5,
    "motorcycle": 0.8,
    "bicycle": 0.7,
    "person": 0.5,
    "tree": 1.0
}
FOCAL_LENGTH = 1000


def estimate_distance(object_width_pixels, object_type):
    if object_type not in KNOWN_WIDTHS:
        return None
    real_width = KNOWN_WIDTHS[object_type]
    distance = (real_width * FOCAL_LENGTH) / object_width_pixels
    return round(distance, 1)


def classify_vehicle(label):
    if label == "car":
        return "4-wheeler"
    elif label == "truck":
        return "6-wheeler"
    elif label in ["motorcycle", "bicycle"]:
        return "2-wheeler"
    return label


def detect_speed_breakers(frame):
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


# Global variables for WebSocket data
obstacles = []
announcements = []


def run_obstacle_detection():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    last_announce_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Speed breaker detection
        speed_breaker_detected = detect_speed_breakers(frame)
        if speed_breaker_detected and time.time() - last_announce_time > 1:
            announcements.append({
                'timestamp': datetime.utcnow().isoformat(),
                'message': 'Speed breaker ahead'
            })
            last_announce_time = time.time()

        # Object detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = object_detector.detect(mp_image)

        # Process detections
        obstacles.clear()
        for detection in detection_result.detections:
            category = detection.categories[0]
            if category.score > 0.5:
                label = category.category_name
                obj_type = classify_vehicle(label)
                distance = estimate_distance(detection.bounding_box.width, label)
                obstacles.append({
                    'label': label,
                    'type': obj_type,
                    'distance': distance,
                    'confidence': category.score
                })
                if time.time() - last_announce_time > 5:
                    message = f"{obj_type} {distance}m" if distance else obj_type
                    announcements.append({
                        'timestamp': datetime.utcnow().isoformat(),
                        'message': message
                    })
                    last_announce_time = time.time()

        # Emit data to WebSocket clients
        socketio.emit('obstacle_data', {
            'obstacles': obstacles,
            'announcements': announcements
        }, namespace='/ws/obstacles')

        socketio.sleep(0.1)

    cap.release()


@socketio.on('connect', namespace='/ws/obstacles')
def handle_connect():
    print('Client connected to WebSocket')


# Start obstacle detection in a separate thread
threading.Thread(target=run_obstacle_detection, daemon=True).start()

if __name__ == '__main__':
    init_db()
    socketio.run(app, debug=True)