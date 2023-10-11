#!/bin/bash

echo "Starting 1"

python flower_experiment_script_complete_unbalanced_serial_check_point_smote.py 100 > log_checkpoint_200_01_smote.log #

sleep 3  # Sleep for 3s to give to the GC

echo "Starting 2"

python flower_experiment_script_complete_unbalanced_serial_check_point_smote.py 200 > log_checkpoint_200_02_smote.log #
