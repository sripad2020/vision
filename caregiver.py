from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import time
import base64
import redis
import json
import eventlet,threading

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


def gen_frames():
    while True:
        try:
            frame_data = redis_client.get('frame')
            if frame_data:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + base64.b64decode(frame_data) + b'\r\n')
            time.sleep(0.1)
        except Exception as e:
            print(f"Caregiver video feed error: {str(e)}")


@app.route('/caregiver')
def caregiver_dashboard():
    return render_template('caregiver_dashboard.html')


@app.route('/caregiver_video_feed')
def caregiver_video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@socketio.on('connect', namespace='/caregiver_data')
def handle_caregiver_connect():
    print('Caregiver WebSocket connected')
    socketio.emit('status', {'message': 'Connected'}, namespace='/caregiver_data')

    def send_data():
        while True:
            try:
                user_data = json.loads(redis_client.get('user_data') or '{}')
                alerts = json.loads(redis_client.get('alerts') or '[]')
                socketio.emit('data', {
                    'user_id': user_data.get('user_id', 'N/A'),
                    'location': user_data.get('location', None),
                    'destination': user_data.get('destination', 'N/A'),
                    'behavior': user_data.get('behavior', 'N/A'),
                    'alerts': alerts
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