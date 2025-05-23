from flask import Flask, render_template, request, redirect, url_for, flash, jsonify,session
import sqlite3,os
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with a secure secret key

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect('users.db',check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )''')
    conn.commit()
    conn.close()

# Call init_db when the app starts
init_db()

@app.route('/')
def index():
    return redirect(url_for('signup'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.json.get('username')
        email = request.json.get('email')
        password = request.json.get('password')

        if not username or not email or not password:
            return jsonify({'message': 'All fields are required'}), 400

        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
            c.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                     (username, email, hashed_password))
            conn.commit()
            conn.close()
            return jsonify({'message': 'Signup successful'}), 201
        except sqlite3.IntegrityError:
            conn.close()
            return jsonify({'message': 'Username or email already exists'}), 400
        except Exception as e:
            conn.close()
            return jsonify({'message': 'An error occurred'}), 500

    # Fetch all users for display
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT username FROM users')
    caretakers = [row[0] for row in c.fetchall()]
    conn.close()
    return render_template('caregiver_signup.html', caretakers=caretakers)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.json.get('username')
        password = request.json.get('password')

        if not username or not password:
            return jsonify({'message': 'Username and password are required'}), 400

        conn = get_db_connection()
        c = conn.cursor()
        c.execute('SELECT password FROM users WHERE username = ?', (username,))
        user = c.fetchone()
        conn.close()

        if user and check_password_hash(user['password'], password):
            from flask import session
            session['username'] = username
            return redirect('/caretakers')
        else:
            return jsonify({'message': 'Invalid username or password'}), 401

    return render_template('caregiver_login.html')

DATABASE = 'users.db'
sqlite3.connect(DATABASE,check_same_thread=False)
def init_db():
    if not os.path.exists(DATABASE):
        conn = sqlite3.connect(DATABASE,check_same_thread=False)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE users_info (
                username TEXT PRIMARY KEY,
                speech_credential TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()
init_db()

def init_dbs():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users_info'")
    if not c.fetchone():
        c.execute('''
            CREATE TABLE users_info (
                username TEXT PRIMARY KEY,
                speech_credential TEXT NOT NULL
            )
        ''')
        conn.commit()
    conn.close()

def get_db_connections():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


init_dbs()

@app.route('/create-caretaker', methods=['GET', 'POST'])
def create_caretaker():
    if 'username' not in session:
        return jsonify({'message': 'Please log in to create a caretaker'}), 401

    if request.method == 'POST':
        username = request.json.get('username')
        speech_credential = request.json.get('speech_credential')
        if not username or not speech_credential:
            return jsonify({'message': 'Username and speech credential are required'}), 400

        conn = get_db_connections()
        c = conn.cursor()
        try:
            c.execute('SELECT username FROM users_info WHERE username = ?', (username,))
            if c.fetchone():
                conn.close()
                return jsonify({'message': 'Username already exists'}), 400

            c.execute('INSERT INTO users_info (username, speech_credential) VALUES (?, ?)',
                      (username, speech_credential))
            conn.commit()
            conn.close()
            return jsonify({'success': True, 'message': 'Caretaker created successfully'}), 201
        except sqlite3.Error as e:
            conn.close()
            return jsonify({'message': f'Database error: {str(e)}'}), 500

    conn = get_db_connections()
    c = conn.cursor()
    c.execute('SELECT username FROM users_info')
    caretakers = [row['username'] for row in c.fetchall()]
    conn.close()
    return render_template('user_SIgnup.html', caretakers=caretakers, current_user=session['username'])

@app.route('/caretakers')
def caretakers():
    if 'username' not in session:
        flash('Please log in to access this page', 'error')
        return redirect(url_for('speech_login'))

    conn = get_db_connections()
    c = conn.cursor()
    c.execute('SELECT username FROM users_info')
    caretakers = [row['username'] for row in c.fetchall()]
    conn.close()
    return render_template('caretaker.html', caretakers=caretakers, current_user=session['username'])


@app.route('/speech-login', methods=['GET', 'POST'])
def speech_login():
    if request.method == 'POST':
        speech_credential = request.json.get('speech_credential')
        if not speech_credential:
            return jsonify({'message': 'Speech credential is required'}), 400

        conn = get_db_connections()
        c = conn.cursor()
        c.execute('SELECT username FROM users_info WHERE speech_credential = ?', (speech_credential,))
        user = c.fetchone()
        conn.close()

        if user:
            session['username'] = user['username']
            return jsonify({'success': True, 'message': 'Login successful'}), 200
        return jsonify({'message': 'Invalid speech credential'}), 401

    return render_template('login.html')


@app.route('/logout')
def logout():
    from flask import session
    session.pop('username', None)
    flash('You have been logged out', 'success')
    return redirect(url_for('login'))


if __name__ == '__main__':
    app.run(debug=True)