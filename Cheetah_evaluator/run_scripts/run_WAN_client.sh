#!/bin/bash

# Number of times to run the commands
num_runs=10

# Net from sqnet, resnet50, densenet121, short1, short2, conv1-9, mp1-9, convmp1-5, ap1-9, convap1-9
for net in sqnet resnet50 densenet121 short1 short2 conv1 conv2 conv3 conv4 conv5 conv6 conv7 conv8 conv9 mp1 mp2 mp3 mp4 mp5 mp6 mp7 mp8 mp9 convmp1 convmp2 convmp3 convmp4 convmp5 ap1 ap2 ap3 ap4 ap5 ap6 ap7 ap8 ap9 convap1 convap2 convap3 convap4 convap5 convap6 convap7 convap8 convap9
do
    # Directory for log files
    log_dir="/home/eloise/cheetah/result-client-WAN/$net"
    echo "Net $net"

    # Loop to run the commands
    for ((i=1; i<=$num_runs; i++))
    do
        echo "Run $i"

        # Create data_$i directory for logs
        mkdir -p "$log_dir/data_$i"

        # Change directory to /home/eloise/eloise/OpenCheetah
        cd /home/eloise/cheetah/OpenCheetah

        # Run client command and redirect output to log file
        bash /home/eloise/cheetah/OpenCheetah/scripts/run-client.sh cheetah $net >> $log_dir/data_$i/log_client.txt &
        client_pid=$!

        # Wait for both server and client commands to finish
        echo "Client PID $client_pid"
        wait $client_pid
        echo "Exit status: $?"

        # Sleep for 5 second before starting the next run
        sleep 5
    done
done