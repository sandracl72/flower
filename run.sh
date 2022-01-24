#!/bin/bash

# Loading script arguments
NBCLIENTS="${1:-3}" # Nb of clients launched by the script (default to 3)
NBMINCLIENTS="${2:-3}" # Nb min of clients before launching round (default to 3)
NBFITCLIENTS="${3:-3}" # Nb of clients sampled for the round (default to 3)
NBROUNDS="${4:-10}" # Nb of rounds (default to 10)
NBEPOCHS="${5:-1}" # Nb of epochs per round (default to 1)

python server_advanced.py -r $NBROUNDS -fc $NBFITCLIENTS -ac $NBMINCLIENTS &
sleep 10 # Sleep for N seconds to give the server enough time to start, increase if clients can't connect

# for ((nb=0; nb<$NBCLIENTS; nb++))  
for i in `seq 0 $(expr $NBCLIENTS - 1)`; do   
    echo "Starting client $i" 
    python client_isic.py --partition=${i} --epochs=1 &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
# If still not stopping you can use `killall python` or `killall python3` or ultimately `pkill python`
sleep 86400