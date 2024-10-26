from hmac import new
from io import StringIO
from unittest import result
from dotenv import load_dotenv
import pandas as pd
from wiguard.dataset import CLIP_SIZE, LABELS
from wiguard.predict import predict
from wiguard.mqtt import client

csi = pd.DataFrame()


def on_message(client, userdata, msg):
    global csi
    message = msg.payload.decode()
    csidata = pd.read_csv(StringIO(message), header=None)
    csi = pd.concat([csi, csidata], ignore_index=True)

    if len(csi) >= CLIP_SIZE:
        result = predict(csi)
        csi = pd.DataFrame()
        print(f"Result: {LABELS[result]}")
        client.publish('wiguard/result', result)


if __name__ == '__main__':
    client.on_message = on_message
    client.loop_forever()
