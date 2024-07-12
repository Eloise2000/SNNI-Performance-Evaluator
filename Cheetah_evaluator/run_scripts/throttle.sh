#!/bin/bash
######
# Taken from https://github.com/emp-toolkit/emp-readme/blob/master/scripts/throttle.sh
######

## replace DEV=lo with your card (e.g., eth0)
DEV=ens3 
if [ "$1" == "del" ]
then
	sudo tc qdisc del dev $DEV root
fi

if [ "$1" == "lan" ]
then
sudo tc qdisc del dev $DEV root
## about 3Gbps
sudo tc qdisc add dev $DEV root handle 1: tbf rate 3000mbit burst 100000 limit 10000
## about 0.3ms ping latency
sudo tc qdisc add dev $DEV parent 1:1 handle 10: netem delay 0.15msec
fi

if [ "$1" == "wan" ]
then
# Check if qdisc with handle zero exists
sudo tc qdisc show dev $DEV | grep -q "qdisc 1: root"
if [ $? -eq 0 ]; then
    # Qdisc exists, so delete it
    sudo tc qdisc del dev $DEV root
else
    echo "Qdisc with handle zero not found. Nothing to delete."
fi

## about 400Mbps
sudo tc qdisc add dev $DEV root handle 1: tbf rate 400mbit burst 100000 limit 10000
## about 10ms ping latency (each 5ms)
sudo tc qdisc add dev $DEV parent 1:1 handle 10: netem delay 5msec
fi
