import asyncio
import os
from threading import Thread
from time import sleep, time
from matplotlib import pyplot as plt
from matplotlib.widgets import Button
import pandas as pd
from io import StringIO
from wiguard.dataset import CLIP_SIZE
from wiguard.show import update
from wiguard.mqtt import client

csi = pd.DataFrame()


csi_path = None


ax_button = plt.axes((0, 0.95, 0.15, 0.05))
button = Button(ax_button, 'Collect CSI')
button.on_clicked(lambda _: Thread(target=collect).start())


def collect():
    global collect_start_time, csi_path
    if csi_path is not None:
        return
    csi_path = os.path.join('data', f'csi-{int(time())}.csv')
    print(f"Collecting CSI to {csi_path}")
    button.label.set_text('Collecting...')
    sleep(5)
    csi_path = None
    button.label.set_text('Collect CSI')


def on_message(client, userdata, msg):
    global csi
    message = msg.payload.decode()
    csidata = pd.read_csv(StringIO(message), header=None)
    csi = pd.concat([csi, csidata], ignore_index=True)

    if csi_path is not None:
        csidata.to_csv(csi_path, index=False, header=False, mode='a')

    if len(csi) >= CLIP_SIZE:
        csi = csi.iloc[-CLIP_SIZE:].reset_index(drop=True)

    update(csi)


if __name__ == '__main__':
    client.on_message = on_message
    client.loop_start()
    plt.show()
