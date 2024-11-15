import os
from threading import Thread
from time import sleep, time
from matplotlib import pyplot as plt
from matplotlib.widgets import Button, Slider
import pandas as pd
from io import StringIO
from wiguard.dataset import CLIP_SIZE
from wiguard.show import update
from wiguard.mqtt import client

# The buffer stores all the CSI data shown on the plot
csi = pd.DataFrame()


collect_duration = 5
csi_path = None


def update_collect_duration(val):
    global collect_duration
    collect_duration = val


def collect_once():
    '''
    Collect CSI data for a certain duration and save it to a file.
    '''
    global csi_path
    if csi_path is not None:
        return
    csi_path = os.path.join('data', f'csi-{int(time())}.csv')
    print(f"Collecting CSI to {csi_path} for {collect_duration}s")
    sleep(collect_duration)
    print(f"Collecting CSI to {csi_path} finished")
    csi_path = None


def on_message(client, userdata, msg):
    '''
    When receiving a message, the message is decoded and stored in the buffer.
    '''
    global csi
    message = msg.payload.decode()
    csidata = pd.read_csv(StringIO(message), header=None)
    csi = pd.concat([csi, csidata], ignore_index=True)

    # If csi_path is not None, save the data to the file
    if csi_path is not None:
        csidata.to_csv(csi_path, index=False, header=False, mode='a')

    if len(csi) >= CLIP_SIZE:
        csi = csi.iloc[-CLIP_SIZE:].reset_index(drop=True)

    update(csi)


button = Button(plt.axes((0, 0.95, 0.15, 0.05)), 'Collect CSI')
button.on_clicked(lambda _: Thread(target=collect_once).start())

slider = Slider(plt.axes((0.35, 0.95, 0.15, 0.05)),
                'Collect Duration', 1, 10, valinit=collect_duration, valstep=1)
slider.on_changed(update_collect_duration)


if __name__ == '__main__':
    client.on_message = on_message
    client.loop_start()
    plt.show()
