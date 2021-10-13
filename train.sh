#!/bin/bash
clear

# training parameters
printf "Training start ...\n\n"

# run the training process
for flag in "BE" "OER" "ORR"
    do
        nohup python -u feature_selection.py $flag >./logs/running_V2_${flag}.log 2>&1 &
    done

exit 0