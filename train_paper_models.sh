#!/bin/bash
clear

# training parameters
printf "Training start ...\n\n"

# run the training process
for flag in "BE" "OER" "ORR"
    do
        nohup python -u train.py $flag >./logs/running_paper_models_${flag}.log 2>&1 &
    done

exit 0