import subprocess
import time
import csiread
import numpy as np
import socketio


send_freq = data['Collection']['send_freq']
collect_freq = data['Collection']['collect_freq']
file_template = data['Collection']['file_template']
disconnect= data['Commands']['disconnect']
reconnect = data['Commands']['reconnect']
log_to_file = data['Commands']['log_to_file']
server_addr = data['Addresses']['server_addr']
wifi_ip = data['Addresses']['wifi_ip']

passwd = "123"
ping = ["ping", wifi_ip, "-i", str(collect_freq), "&"]
save_path = "./csidata/"
sio = socketio.Client()
sio.connect(server_addr)
@sio.event
def connect():
    print('Connected to server')
@sio.event
def disconnect():
    print('Disconnected from server')

disconnect_res = subprocess.run(disconnect, input=passwd, shell=False, capture_output=True, text=True)
time.sleep(3)
reconnect_res = subprocess.run(reconnect, input=passwd, shell=False, capture_output=True, text=True)
time.sleep(7)
ping_process = subprocess.Popen(ping, shell=False)
try:
    while True:
        current_time = time.strftime("%m%d_%H_%M_%S", time.localtime())
        file_name = save_path + file_template + current_time + ".dat"
        log_to_file.append(file_name)
        log_to_file.append("&")
        log_to_file_process = subprocess.Popen(log_to_file, shell=False)

        time.sleep(send_freq)
        log_to_file_process.terminate()

        csidata = csiread.Intel(file_name)
        csidata.read()
        scaled_csi: np.ndarray = csidata.get_scaled_csi()
        sio.emit('csi', scaled_csi)
except:
    ping_process.terminate()
    log_to_file_process.terminate()
    sio.disconnect()

ping_process.terminate()
sio.disconnect()
    