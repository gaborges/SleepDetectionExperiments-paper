#!/bin/bash

echo "Starting server"
python Server_ISABELA.py > server.log & #

sleep 3  # Sleep for 3s to give the server enough time to start

for i in {0..18}
do
    echo "Starting client $i"
    python Client_ISABELA.py $i &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
