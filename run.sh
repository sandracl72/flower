#!/bin/bash

# Loading script arguments

while getopts ":nc:ac:fc:r:e:p:" flag; do
    case "${flag}" in
        nc) NBCLIENTS=${OPTARG:-5};;   # Nb of clients launched by the script (default to 5)
        ac) NBMINCLIENTS=${OPTARG:-5};;  # Nb min of clients before launching round (default to 5)
        fc) NBFITCLIENTS=${OPTARG:-5};;  # Nb of clients sampled for the round (default to 5)
        r) NBROUNDS=${OPTARG:-10};;  # Nb of rounds (default to 10)
        e) NBEPOCHS=${OPTARG:-2};;  # Nb of epochs per round (default to 2)
        p) PATH=${OPTARG};;
    esac
done

python server_advanced_mp.py -r $NBROUNDS -fc $NBFITCLIENTS -ac $NBMINCLIENTS --path $PATH&
sleep 10 # Sleep for N seconds to give the server enough time to start, increase if clients can't connect

# for ((nb=0; nb<$NBCLIENTS; nb++))  
for i in `seq 0 $(expr $NBCLIENTS - 1)`; do   
    echo "Starting client $i" 
    python client_isic_mp.py --partition=${i} --epochs=$NBEPOCHS --path=$PATH &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
# If still not stopping you can use `killall python` or `killall python3` or ultimately `pkill python`
sleep 86400