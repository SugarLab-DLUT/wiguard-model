from wifall import config
import socketio

sio = socketio.Client()
sio.connect(config['server'])
@sio.event
def connect():
    print('Connected to server')
@sio.event
def disconnect():
    print('Disconnected from server')