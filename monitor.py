import csiread
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

import subprocess
import time
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

send_freq = config.getfloat('Network', 'send_freq')
collect_freq = config.getfloat('Network', 'collect_freq')
gateway_ip = config.get('Network', 'gateway_ip')
disconnect = config.get('Commands', 'disconnect').split(', ')
reconnect = config.get('Commands', 'reconnect').split(', ')
log_to_file = config.get('Commands', 'log_to_file')
csidata_path = 'dat/csi-1701504722.dat'

rx_num = 3
tx_num = 2

# print('disconnecting')
# subprocess.run(disconnect)
# time.sleep(5)
# print('reconnecting')
# subprocess.run(reconnect)
# time.sleep(10)
# subprocess.Popen(['ping', gateway_ip, '-i', str(collect_freq)])

# while True:
#     log_process = subprocess.Popen([log_to_file, csidata_path])
#     time.sleep(send_freq)
#     log_process.kill()
#     csidata = csiread.Intel(csidata_path)
#     draw(csidata.get_scaled_csi())
csidata = csiread.Intel(csidata_path)
csidata.read()
csi = csidata.get_scaled_csi()

fig, axs = plt.subplots(tx_num+1, 2, figsize=(14, 6))
fig.subplots_adjust(hspace=0.7, wspace=0.2)
x = np.arange(30)

for tx in range(tx_num):
    axs[tx, 0].set_title(f'tx ant{tx}')
    axs[tx, 0].set_xlabel('subcarriers')
    axs[tx, 0].set_ylabel('amplitude')
    axs[tx, 1].set_title(f'tx ant{tx}')
    axs[tx, 1].set_xlabel('subcarriers')
    axs[tx, 1].set_ylabel('phase')

amplis = [[axs[tx, 0].plot(x, csi[0, :, rx, tx].real, label=f'rx ant{rx}')[0]
           for rx in range(rx_num)]for tx in range(tx_num)]
phases = [[axs[tx, 1].plot(x, csi[0, :, rx, tx].imag, label=f'rx ant{rx}')[0]
           for rx in range(rx_num)]for tx in range(tx_num)]

axs[tx_num, 0].set_xlabel('packets')
axs[tx_num, 0].set_ylabel('amplitude')
for rx in range(rx_num):
    axs[tx_num, 0].plot(csi[:, 0, rx, 0].real, label=f'rx ant{rx}')

axs[tx_num, 1].set_xlabel('packets')
axs[tx_num, 1].set_ylabel('phase')
for rx in range(rx_num):
    axs[tx_num, 1].plot(csi[:, 0, rx, 0].imag, label=f'rx ant{rx}')


def animate(i):
    for tx in range(tx_num):
        for rx in range(rx_num):
            amplis[tx][rx].set_ydata(csi[i, :, rx, tx].real)
            phases[tx][rx].set_ydata(csi[i, :, rx, tx].imag)
    return [item for sublist in amplis for item in sublist] + [item for sublist in phases for item in sublist]


ani = animation.FuncAnimation(
    fig, animate, interval=200, blit=True)

fig.legend()
plt.show()
