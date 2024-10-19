# WiGuard - Model

Wireless sensing based health monitoring system.

## Workflow

```mermaid
graph LR

Collector-->Broker
Model<-->Broker
Broker-->Nuxt.js
```

## How to run

First, create a `.env` file in the root directory of the project and add the following content:

```properties
BROKER = 'broker.emqx.io'
PORT = 1883
TOPIC_SUB = 'wiguard/original_csi'
TOPIC_PUB = 'wiguard/after_test_csi'
BUFFER_SIZE = 110
```

Then, install the dependencies and run the application:

```bash
pip install -r requirements.txt
python main.py
```

## Utils

**Visualize csi data**

```bash
python -m wiguard.show csidata.csv
```
