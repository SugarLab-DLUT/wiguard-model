#!/bin/bash

echo disconnecting
sudo modprobe -r iwlwifi mac80211
sleep 5s
echo reconnecting
sudo modprobe iwlwifi connector_log=0x1
sleep 10s
ping 192.168.3.1 -i 0.2 &

while true
do
~/linux-80211n-csitool-supplementary/netlink/log_to_file csi-temp.dat &
sleep 5s
kill $!
cat csi-temp.dat
cat csi-temp.dat > /dev/tcp/192.168.3.35/1212
done
