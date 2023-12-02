from socket import *
from time import time

server = socket(AF_INET, SOCK_STREAM)

server.bind(('0.0.0.0', 1212))
server.listen()

while True:
    client, addr = server.accept()
    t = int(time())
    data = b''
    while True:
        packet = client.recv(1024)
        if not packet:
            break
        data += packet
    print(f'recv package {t} ({len(data)} bytes)')
    with open(f'dat/csi-{t}.dat', 'wb') as f:
        f.write(data)
