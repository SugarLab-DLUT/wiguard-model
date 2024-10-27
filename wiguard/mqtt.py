import os
from pydoc import cli
import random
from matplotlib import pyplot as plt
import pandas as pd
import paho.mqtt.client as mqtt
from io import StringIO
from dotenv import load_dotenv
from paho.mqtt.enums import CallbackAPIVersion
from wiguard.dataset import CLIP_SIZE
from wiguard.show import update


load_dotenv()
broker_host = os.getenv('MQTT_BROKER_HOST') or 'localhost'
broker_port = int(os.getenv('MQTT_BROKER_PORT') or 1883)
client_id = f'python-mqtt-{random.randint(0, 1000)}'

csi = pd.DataFrame()


def on_message(client, userdata, msg):
    global csi
    message = msg.payload.decode()
    csidata = pd.read_csv(StringIO(message), header=None)
    csi = pd.concat([csi, csidata], ignore_index=True)

    if len(csi) >= CLIP_SIZE:
        csi = csi.iloc[-CLIP_SIZE:].reset_index(drop=True)

    update(csi)


print(f"Connecting to mqtt://{broker_host}:{broker_port}")
client = mqtt.Client(CallbackAPIVersion.VERSION2)
client.connect(broker_host, broker_port)
client.subscribe('wiguard/csi')


if __name__ == '__main__':
    client.on_message = on_message
    client.loop_start()
    plt.show()
    client.loop_stop()
