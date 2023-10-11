#!/bin/bash

python 03_federated_LSTM_unbalanced_1_epoch.py > fed_lstm01_unb_ep_01.log  #
python 03_federated_LSTM_unbalanced_3_epoch.py > fed_lstm01_unb_ep_03.log  #
python 03_federated_LSTM_unbalanced_5_epoch.py > fed_lstm01_unb_ep_05.log  #

python 03_federated_MLP_unbalanced_1_epoch.py > fed_mlp01_unb_ep_01.log 
python 03_federated_MLP_unbalanced_3_epoch.py > fed_mlp01_unb_ep_03.log 
python 03_federated_MLP_unbalanced_5_epoch.py > fed_mlp01_unb_ep_05.log 

python 00_LSTM_unbalanced_script_4_time_steps.py > trad_lstm_umb_200_epc_4_t.log
python 00_LSTM_unbalanced_script_2_time_steps.py > trad_lstm_umb_200_epc_2_t.log
python 00_MLP_unbalanced_script.py > trad_mlp_umb_200_epc.log

python 01_LSTM_BI_unbalanced_script_2_time_steps.py > trad_lstm_bi_umb_200_epc_4_t.log
python 01_LSTM_BI_unbalanced_script_4_time_steps.py > trad_lstm_bi_umb_200_epc_2_t.log
