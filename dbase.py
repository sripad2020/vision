import sqlite3

conn = sqlite3.connect('users.db')
c = conn.cursor()


c.execute('DROP TABLE users_info')
c.execute('CREATE TABLE users_info (username TEXT PRIMARY KEY, speech_credential TEXT NOT NULL, caregiver_username TEXT NOT NULL)')
conn.commit()
conn.close()