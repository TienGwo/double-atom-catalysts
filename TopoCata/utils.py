# -*- coding: utf-8 -*-

import os
import shutil
import datetime


def mkdir(path, clear=False):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    
    if isExists:
        if clear:
            shutil.rmtree(path)
            os.makedirs(path)
        else:
            pass
        
    else:
        os.makedirs(path)
        
        
def print_all_info(sample_path, filename, lr, use_list):
    now_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    f = open(filename, "a")
    
    print(" \n================= Training =================", file=f)
    print("TRAINING START:", file=f)
    print("Samples are in", sample_path, "\n", file=f)
    print("    START TIME:", now_time, "(EST)", file=f)
    print("    Learning rate:", f"{lr:.1e}", file=f)
    print("    Use list:", use_list, file=f)
    print("", file=f)
    
    f.close()
    
    
def print_monitor(epoch, lr, MSE_train, MSE_valid, R2_train, R2_valid, MAPE_train, MAPE_valid, filename):
    f = open(filename, "a")
    print(
        f"{epoch:03d}", "===",
        f"{lr:.3e}", "===",
        "MSE [", f"{MSE_train:.3f}" + ",", f"{MSE_valid:.3f}", "] ===",
        "R2 [", f"{R2_train:.3f}" + ",", f"{R2_valid:.3f}", "] ===",
        "MAPE [", f"{MAPE_train:.3f}" + "%" + ",", f"{MAPE_valid:.3f}" + "%", "]",
        file=f
        )
    
    f.close()