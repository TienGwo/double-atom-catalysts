# -*- coding: utf-8 -*-

import sys
from feature_selection import main
from TopoCata.train.train_func import setup_seed
from TopoCata.utils import mkdir

if __name__ == "__main__":
    setup_seed(seed=1024)
    sample_path="./dataset/"
    
    flag = sys.argv[1]    
    
    if flag == "BE":
        use_list = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]
    elif flag == "OER":
        use_list = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]
    else:
        use_list = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10]

    for times in range(10):
        logfile = "./model_save_paper/training_log_" + flag + ".log"

        f = open(logfile, "a")
        print("\n", "=" * 10, times, "=" * 10, "\n", file=f)
        f.close()

        model_save_path = "./model_save_paper/" + flag + "/" + str(times) + "/"
        mkdir(model_save_path, clear=True)
        main(flag, use_list, sample_path, model_save_path, logfile, times)

    