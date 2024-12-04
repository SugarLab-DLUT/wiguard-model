from io import StringIO
import pandas as pd
import serial
import serial.tools.list_ports

from wiguard.predict import predict
from wiguard.dataset import CLIP_SIZE, LABELS


csi = pd.DataFrame()


ports = list(serial.tools.list_ports.comports())
if len(ports) == 0:
    print("No serial ports found")
    exit(1)
print(f"Using port: {ports[0].device}")


if __name__ == '__main__':
    with serial.Serial(ports[0].device, 115200) as io:
        while True:
            try:
                line = io.readline().decode()
                if line.startswith("CSI_DATA"):
                    csidata = pd.read_csv(StringIO(line), header=None)
                    csi = pd.concat([csi, csidata], ignore_index=True)
                    print('.', end='', flush=True)
                else:
                    print(line, end='')

                if len(csi) >= CLIP_SIZE:
                    print(len(csi))
                    result = predict(csi)
                    csi = pd.DataFrame()
                    print(f"Result: {LABELS[result]}")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                continue
