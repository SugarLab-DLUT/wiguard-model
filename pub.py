import random
import time
from paho.mqtt import client as mqtt_client

broker = 'broker.emqx.io'
port = 1883               
topic = "wiguard/original_csi"    
# generate client ID with pub prefix randomly
client_id = f'python-mqtt-{random.randint(0, 1000)}'


def connect_mqtt():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id=client_id)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client


def publish(client):
    msg_count = 0
    with open('.../data.txt', 'r') as f:
        for line in f:
            time.sleep(1)
            msg = line.strip()

            result = client.publish(topic, msg)
            status = result[0]
            if status == 0:
                print(f"Send `{msg}` to topic `{topic}`")
            else:
                print(f"Failed to send message to topic {topic}")
            msg_count += 1



def run():
    client = connect_mqtt()
    client.loop_start()
    publish(client)


if __name__ == '__main__':
    run()
