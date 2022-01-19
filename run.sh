#!/bin/bash

python server_advanced.py &
sleep 10 # Sleep for 10s to give the server enough time to start

for i in `seq 0 3`; do
    echo "Starting client $i"
    python client_isic.py --partition=${i} --num_partitions=5 &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait