from hmac import new
from dotenv import load_dotenv
import random
import paho.mqtt.client as mqtt
import json
from wiguard.test import test_predict
import os
import csv

"""
subsribe csi data 
publish after-test data
"""
load_dotenv()
broker = os.getenv('BROKER')
port = int(os.getenv('PORT'))
topic_sub = os.getenv('TOPIC_SUB')
topic_pub = os.getenv('TOPIC_PUB')
buffer_size = int(os.getenv('BUFFER_SIZE')) 
message_list = []

client_id = f'python-mqtt-{random.randint(0, 1000)}'

def connect_mqtt() -> mqtt.Client:
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print(f"Failed to connect, return code {rc}\n")

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1,client_id)
    client.on_connect = on_connect
    try:
        client.connect(broker, port)
    except Exception as e:
        print(f"Connection failed: {e}")
    return client

def subscribe(client: mqtt.Client):
    def on_message(client, userdata, msg):
        global   message_list
        print(f"{len(message_list)}Received from `{msg.topic}` topic")
        message = msg.payload.decode()  

        if len(message) > 0 :
            first_pos = message.find('[')
            message_list.append(message[first_pos:-1])

        if len(message_list) >= buffer_size:
            print(message_list)
            test_result = test_predict(message_list)
            message_list.clear()
            client.publish(topic_pub, json.dumps(test_result), qos=1)
            print(f"Test result: {test_result}")

    client.subscribe(topic_sub)
    client.on_message = on_message


if __name__ == '__main__':
    client = connect_mqtt()
    subscribe(client)
    client.loop_forever()