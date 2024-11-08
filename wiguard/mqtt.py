import os
import random
import paho.mqtt.client as mqtt
from dotenv import load_dotenv
from paho.mqtt.enums import CallbackAPIVersion


load_dotenv()
broker_host = os.getenv('MQTT_BROKER_HOST') or 'localhost'
broker_port = int(os.getenv('MQTT_BROKER_PORT') or 1883)
client_id = f'python-mqtt-{random.randint(0, 1000)}'


print(f"Connecting to mqtt://{broker_host}:{broker_port}")
client = mqtt.Client(CallbackAPIVersion.VERSION2)
client.connect(broker_host, broker_port)
client.subscribe('wiguard/csi')
