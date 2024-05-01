## summary2

### 采集.dat改由python实现，自动重命名功能

```python
disconnect_res = subprocess.run(disconnect, input=passwd, shell=False, capture_output=True, text=True)
time.sleep(3)
reconnect_res = subprocess.run(reconnect, input=passwd, shell=False, capture_output=True, text=True)
time.sleep(7)
ping_process = subprocess.Popen(ping, shell=False)
try:
    while True:
        current_time = time.strftime("%m%d_%H_%M_%S", time.localtime())
        file_name = save_path + file_template + current_time + ".dat"
        log_to_file.append(file_name)
        log_to_file.append("&")
        log_to_file_process = subprocess.Popen(log_to_file, shell=False)
```

### 使用toml库管理配置文件

```toml
[Collection]
send_freq = 5
collect_freq = 0.2
file_template = "csi_"

[Commands]
disconnect = ["sudo", "modprobe", "-r", "iwlwifi", "mac80211"]
reconnect = ["sudo", "modprobe", "iwlwifi", "connector_log=0x1"]
log_to_file = ["~/linux-80211n-csitool-supplementary/netlink/log_to_file"]

[Addresses]
server_addr = "wiguard.saurlax.com:3000"
wifi_ip = "192.168.3.1"
```

### socketio连接服务端被控

```python
sio = socketio.Client()
sio.connect(server_addr)
@sio.event
def connect():
    print('Connected to server')
@sio.event
def disconnect():
    print('Disconnected from server')


csidata = csiread.Intel(file_name)
csidata.read()
scaled_csi: np.ndarray = csidata.get_scaled_csi()
sio.emit('csi', scaled_csi)
```

