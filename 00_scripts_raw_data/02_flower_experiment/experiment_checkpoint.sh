#!/bin/bash

echo "Starting 1"

python flower_experiment_script_complete_unbalanced_serial_check_point.py 100 > log_checkpoint_200_01.log #

sleep 3  # Sleep for 3s to give to the GC

echo "Starting 2"

python flower_experiment_script_complete_unbalanced_serial_check_point.py 200 > log_checkpoint_200_02.log #
