from flask import Flask, render_template
from flask_socketio import SocketIO
import time
import redis
import json
import eventlet,threading
import glob
import os

# Patch eventlet
eventlet.monkey_patch()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(
    app,
    ping_timeout=20,
    ping_interval=10,
    async_mode='eventlet',
    cors_allowed_origins=['http://localhost:5001', 'http://127.0.0.1:5001']
)
redis_client = redis.Redis(host='localhost', port=6379, db=0)


def get_latest_log_file():
    log_files = glob.glob("caregiver_log_*.txt")
    if not log_files:
        return None
    return max(log_files, key=os.path.getmtime)


@app.route('/caregiver')
def caregiver_dashboard():
    return render_template('caregiver_dashboard.html')


@socketio.on('connect', namespace='/caregiver_data')
def handle_caregiver_connect():
    print('Caregiver WebSocket connected')
    socketio.emit('status', {'message': 'Connected'}, namespace='/caregiver_data')

    def send_data():
        last_log_position = 0
        while True:
            try:
                # Get user data and alerts
                user_data = json.loads(redis_client.get('user_data') or '{}')
                alerts = json.loads(redis_client.get('alerts') or '[]')

                # Read latest log file
                log_file = get_latest_log_file()
                log_entries = []
                if log_file:
                    with open(log_file, 'r') as f:
                        f.seek(last_log_position)
                        new_logs = f.readlines()
                        log_entries = [line.strip() for line in new_logs]
                        last_log_position = f.tell()

                socketio.emit('data', {
                    'user_id': user_data.get('user_id', 'N/A'),
                    'location': user_data.get('location', None),
                    'destination': user_data.get('destination', 'N/A'),
                    'behavior': user_data.get('behavior', 'N/A'),
                    'alerts': alerts,
                    'logs': log_entries
                }, namespace='/caregiver_data')
                time.sleep(1)
            except Exception as e:
                print(f"Caregiver data error: {str(e)}")

    threading.Thread(target=send_data, daemon=True).start()


@socketio.on('disconnect', namespace='/caregiver_data')
def handle_disconnect():
    print('Caregiver WebSocket disconnected')


@socketio.on_error(namespace='/caregiver_data')
def handle_error(e):
    print(f"Caregiver WebSocket error: {str(e)}")


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5001, allow_unsafe_werkzeug=True)