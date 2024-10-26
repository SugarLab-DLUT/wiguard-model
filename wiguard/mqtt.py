import os
import random
from dotenv import load_dotenv
import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion


load_dotenv()
broker_host = os.getenv('MQTT_BROKER_HOST') or 'localhost'
broker_port = int(os.getenv('MQTT_BROKER_PORT') or 1883)
client_id = f'python-mqtt-{random.randint(0, 1000)}'


def on_message(client, userdata, msg):
    print(f"{msg.topic}: {msg.payload.decode()}")


print(f"Connecting to {broker_host}:{broker_port}")
client = mqtt.Client(CallbackAPIVersion.VERSION2)
client.connect(broker_host, broker_port)
client.subscribe('wiguard/csi')


if __name__ == '__main__':
    client.on_message = on_message
    client.loop_forever()
