#!/bin/bash

python 03_federated_LSTM_oversample-smote_1_epoch.py > fed_lstm01_ep_01.log  #
python 03_federated_LSTM_oversample-smote_3_epoch.py > fed_lstm01_ep_03.log  #
python 03_federated_LSTM_oversample-smote_5_epoch.py > fed_lstm01_ep_05.log  #

python 03_federated_MLP_oversample-smote_1_epoch.py > fed_mlp01_ep_01.log 
python 03_federated_MLP_oversample-smote_3_epoch.py > fed_mlp01_ep_03.log 
python 03_federated_MLP_oversample-smote_5_epoch.py > fed_mlp01_ep_05.log 

python 00_LSTM_smote_oversample_script_4_time_steps.py > trad_lstm_smote_200_epc_4_t.log
python 00_LSTM_smote_oversample_script_2_time_steps.py > trad_lstm_smote_200_epc_2_t.log
python 00_MLP_smote_oversample_script.py > trad_mlp_smote_200_epc.log
