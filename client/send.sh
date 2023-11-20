#!/bin/bash

send_delay=5s
packet_delay=0.2s
server_ip='192.168.3.35'
gateway_ip='192.168.3.1'
disconnect_cmd='sudo modprobe -r iwlwifi mac80211'
reconnect_cmd='sudo modprobe iwlwifi connector_log=0x1'
log_to_file_path='~/linux-80211n-csitool-supplementary/netlink/log_to_file'

echo disconnecting
$disconnect_cmd
sleep 5s
echo reconnecting
$reconnect_cmd
sleep 10s
ping $gateway_ip -i $packet_delay &

while true
do
$log_to_file_path &
sleep $delay
kill $!
cat csi-temp.dat
cat csi-temp.dat > /dev/tcp/$server_ip/1212
done
