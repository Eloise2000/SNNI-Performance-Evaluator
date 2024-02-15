#!/bin/bash

# Run as server

# Number of times to run the commands
num_runs=15

# Net from sqnet, resnet50, densenet121, short1, short2, conv1-9, mp1-9, convmp1-5, ap1-9, convap1-9
# net="mp9"

for net in mp3 mp4 mp5 mp6 mp7 mp8 mp9 convmp1 convmp2 convmp3 convmp4 convmp5 ap1 ap2 ap3 ap4 ap5 ap6 ap7 ap8 ap9 convap1 convap2 convap3 convap4 convap5 convap6 convap7 convap8 convap9
do
    # Directory for log files
    log_dir="/home/eloise/eloise/result/$net"
    echo "Net $net"

    # Loop to run the commands
    for ((i=1; i<=$num_runs; i++))
    do
        echo "Run $i"

        # Create data_$i directory for logs
        mkdir -p "$log_dir/data_$i"

        # Change directory to /home/eloise/eloise/OpenCheetah
        cd /home/eloise/eloise/OpenCheetah

        # Run server command and redirect output to log file
        bash /home/eloise/eloise/OpenCheetah/scripts/run-server.sh cheetah $net >> $log_dir/data_$i/log_server.txt &
        server_pid=$!

        # Wait for both server and client commands to finish
        echo "Server PID $server_pid"
        wait $server_pid
        echo "Exit status: $?"

        # Sleep for 5 second before starting the next run
        sleep 5
    done
done