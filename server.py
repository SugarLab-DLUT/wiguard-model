from socket import *
from time import time

server = socket(AF_INET, SOCK_STREAM)

server.bind(('0.0.0.0', 1212))
server.listen()

while True:
    client, addr = server.accept()
    t = int(time())
    data = client.recv(1024)
    print(f'recv package {t} ({data.__len__()} bytes)')
    # TODO
    # This will accumulate more and more unnecessary files.
    # After accessing the model, only input values should be
    # processed each time and no files should be saved.
    with open(f'dat/csi-{t}.dat', 'wb') as f:
        f.write(data)
