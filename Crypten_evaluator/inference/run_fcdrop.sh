#!/bin/bash
. /home/eloise/crypten/script/common.sh
echo $MASTER_ADDR $MASTER_PORT

num_runs=5
# seed options: 17 36 42 49
for seed in 17 36 42 49
do
    net="fcdrop_rs$seed"
    log_dir="/home/eloise/crypten/result_32GB_RAM_16_core/result_server_WAN/$net"
    echo "Net $net"

    # Loop to run the commands
    for ((i=1; i<=$num_runs; i++))
    do
        echo "Run $i"

        # Create data_$i directory for logs
        mkdir -p "$log_dir/data_$i"

        cd /home/eloise/crypten/script/model_run
        # Run server command and redirect output to log file
        python3 fcdrop_run.py $seed >> $log_dir/data_$i/log_server.txt &
        # python3 convmp_run.py $seed &
        server_pid=$!

        # Wait for both server and client commands to finish
        echo "Server PID $server_pid"
        wait $server_pid
        echo "Exit status: $?"

        # Sleep for 5 second before starting the next run
        sleep 5
    done
done