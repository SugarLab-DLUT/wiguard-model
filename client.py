import subprocess
import time
import socket
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

send_freq = config.getfloat('Network', 'send_freq')
collect_freq = config.getfloat('Network', 'collect_freq')
server_ip = config.get('Network', 'server_ip')
gateway_ip = config.get('Network', 'gateway_ip')
disconnect = config.get('Commands', 'disconnect').split(', ')
reconnect = config.get('Commands', 'reconnect').split(', ')
log_to_file = config.get('Commands', 'log_to_file')
csidata_path = 'dat/csi-temp.dat'

print('disconnecting')
subprocess.run(disconnect)
time.sleep(5)
print('reconnecting')
subprocess.run(reconnect)
time.sleep(10)
subprocess.Popen(['ping', gateway_ip, '-i', str(collect_freq)])

while True:
    log_process = subprocess.Popen([log_to_file, csidata_path])
    time.sleep(send_freq)
    log_process.kill()
    with open(csidata_path, 'rb') as file:
        data = file.read()
    print(f'sent at {time.time()}')
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((server_ip, 1212))
    sock.sendall(data)
    sock.close()
