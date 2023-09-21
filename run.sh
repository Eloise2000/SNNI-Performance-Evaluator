#!/bin/bash

# Function to run a command in a new terminal window
function run_in_terminal {
    $SHELL -c "$1" &
    # wait $!
}

# Number of times to run the commands
num_runs=15

# Net from sqnet, resnet50, densenet121, short1, short2, mp2, conv1-9, mp1-9, convmp1-5, ap1-9, convap1-9
# net="mp9"

for net in convap1 convap2 convap3 convap4 convap5 convap6 convap7 convap8 convap9
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

        # Run server command in a new terminal and redirect output to log file
        run_in_terminal "bash /home/eloise/eloise/OpenCheetah/scripts/run-server.sh cheetah $net >> $log_dir/data_$i/log_server.txt"

        # Run client command in a new terminal and redirect output to log file
        run_in_terminal "bash /home/eloise/eloise/OpenCheetah/scripts/run-client.sh cheetah $net >> $log_dir/data_$i/log_client.txt"

        # Wait for both server and client commands to finish
        wait

        # Sleep for 1 second before starting the next run
        sleep 1
    done
done