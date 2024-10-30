# WiGuard - Model

Wireless sensing based health monitoring system.

## How to run

If you use a virtual environment, name it `.venv` or `.conda` as it is already added to the `.gitignore` file.

Then, create a `.env` file in the root directory of the project and add the following content:

```properties
MQTT_BROKER_HOST=localhost
MQTT_BROKER_PORT=1883
```

Install the dependencies and run the application:

```bash
pip install -r requirements.txt
python main.py
```

To run the project on a GPU, you need to install the correct version of PyTorch. For example, to install PyTorch 2.5.0 for CUDA 12.4, run the following command:

```bash
pip3 install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu124
```

## Workflow

```mermaid
graph LR

ESP32-->NanoMQ
Intel5300-->NanoMQ
NanoMQ-->Cloud
Model[⭐Model]-->Cloud
Cloud-->Web
Cloud-->Client
```

## Utils

Visualize csi data:

```bash
python -m wiguard.show csidata.csv
```

Predict one csi file:

```bash
python -m wiguard.predict csidata.csv
```

Subscribe to the MQTT broker and show:

```bash
python -m wiguard.mqtt
```

## APIs
