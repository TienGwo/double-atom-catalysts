# -*- coding: utf-8 -*-

import sys
import torch
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from TopoCata.utils import print_all_info, print_monitor, mkdir
from TopoCata.data.data_loader import build_dataset
from TopoCata.model.model_graph import TopoCata
from TopoCata.train.train_func import setup_seed

model_save_dir = "./model_save_V2/"

def MAPE_calc(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    
def train(model, dataset, loss_func, optimizer):
    model.train()
    for samples in dataset: # Iterate over each mini-batch.
        inputs, labels = samples
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = model(inputs) # Perform a single forward pass.
        loss = loss_func(outputs, labels)
        optimizer.zero_grad() # Clear gradients.
        loss.backward() # Derive gradients.
        optimizer.step() # Update parameters based on gradients.
    return model
    
    
@torch.no_grad()
def valid(model, flag, use_list, target, sample_path):
    inputs = np.load(sample_path + "inputs_" + target + "_" + flag + ".npy")[:, use_list, :]
    labels = np.load(sample_path + "labels_" + target + "_" + flag + ".npy")
    inputs = torch.tensor(np.transpose(inputs, axes=(0, 2, 1))).cuda().float()
  
    model.eval()
    pred = model(inputs).cpu().detach().numpy()
    R2 = r2_score(y_true=labels, y_pred=pred)
    MSE = mean_squared_error(y_true=labels, y_pred=pred)
    MAPE = MAPE_calc(y_true=labels, y_pred=pred)
    
    return R2, MSE, MAPE


def define_list(remove_list):
    
    # define the inputs list
    original_list = [_ for _ in range(11)]
    if remove_list != []:
        for element in remove_list:
            original_list.remove(element)
    
    # define the parameter combination
    inputs_list = []
    for loopi in range(len(original_list) + 1):
        if loopi == 0:
            inputs_list.append(original_list)
        else:
            inputs_list.append(
                [_ for _ in original_list if _ is not original_list[loopi - 1]]
                )
        
    return inputs_list


def eval_epoch(flag, inputs_list, epoch, sample_path):
    with torch.no_grad():
        R2_valid_record = []
        for list_num, use_list in enumerate(inputs_list):
            _, dataset_valid = build_dataset(sample_path, flag, use_list=use_list)
            R2_valid = 0
            for index in range(10):
                model_path = model_save_dir + flag + "/" + str(epoch) + "/" + str(list_num) + "/" + str(index) + "/"
                model = TopoCata(feature_num=len(use_list)).cuda()
                model.load_state_dict(torch.load(model_path + "model.pkl"))
                R2_valid_temp, _, _ = valid(model, flag, use_list, "valid", sample_path)
                R2_valid += R2_valid_temp / 10
                
            R2_valid_record.append(R2_valid)
        
    remove_index = np.where(R2_valid_record == max(R2_valid_record))[0][0] - 1
    remove_item = inputs_list[0][remove_index]
    
    return remove_item


def main(flag, use_list, sample_path, model_save_path, logfile, times):
    epoch_num = 800
    dataset_train, dataset_valid = build_dataset(sample_path, flag, use_list=use_list)
    model = TopoCata(feature_num=len(use_list)).cuda()
    loss_func = torch.nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=200,
        gamma=0.5
        )

    MSE_record = np.zeros([epoch_num, 2], dtype=np.float32)
    MAPE_record = np.zeros_like(MSE_record)
    R2_record = np.zeros_like(MSE_record)
    
    R2_flag = 0
    R2_train_flag, R2_valid_flag = 0, 0
    MAPE_train_flag, MAPE_valid_flag = 0, 0
    
    for epoch in range(epoch_num):
        model = train(model, dataset_train, loss_func, optimizer) # training
        R2_train, MSE_train, MAPE_train = valid(model, flag, use_list, "train", sample_path)
        R2_valid, MSE_valid, MAPE_valid = valid(model, flag, use_list, "valid", sample_path)

        # recording
        MSE_record[epoch, 0] = MSE_train
        MSE_record[epoch, 1] = MSE_valid
        MAPE_record[epoch, 0] = MAPE_train
        MAPE_record[epoch, 1] = MAPE_valid
        R2_record[epoch, 0] = R2_train
        R2_record[epoch, 1] = R2_valid

        if R2_valid > R2_flag and R2_valid < R2_train:
            R2_flag = R2_valid
            R2_train_flag, R2_valid_flag = R2_train, R2_valid
            MAPE_train_flag, MAPE_valid_flag = MAPE_train, MAPE_valid
            
            # model saving
            torch.save(
                obj = model.state_dict(),
                f = model_save_path + "model.pkl"
                )

            lr = optimizer.param_groups[0]["lr"]
            print_monitor(
                epoch, lr, 
                MSE_train, MSE_valid, 
                R2_train, R2_valid,
                MAPE_train, MAPE_valid, 
                logfile
                )
    
    # saving the record
    print("[", times, "]", "===  R2:", "[", f"{R2_train_flag:.3f}" + ",", f"{R2_valid_flag:.3f}", "]", \
          "===  MAPE:", "[", f"{MAPE_train_flag:.3f}" + "%" + ",", f"{MAPE_valid_flag:.3f}" + "%", "]")
    np.save(model_save_path + "MSE_record.npy", MSE_record)
    np.save(model_save_path + "MAPE_record.npy", MAPE_record)
    np.save(model_save_path + "R2_record.npy", R2_record)
    
    
if __name__ == "__main__":
    setup_seed(seed=1024)
    flag = sys.argv[1]
    sample_path = "./dataset/"
    
    remove_list = []
    for epoch in range(10):
        print("=" * 10, epoch, "=" * 10)
        inputs_list = define_list(remove_list)
        
        # define the parameter combination
        for feature_index, use_list in enumerate(inputs_list):
            print("\n", use_list)
            model_save_path = model_save_dir + flag + "/" + str(epoch) + "/" + str(feature_index) + "/"
            mkdir(model_save_path, clear=True)
            np.savetxt(model_save_path + "use_list.txt", use_list, fmt="%d")
            logfile = model_save_path + "training_log.txt"

            # training
            for times in range(10):
                print_all_info(
                    sample_path=sample_path,
                    filename=logfile,
                    lr=1.0e-3,
                    use_list=use_list
                    )

                model_save_path_times = model_save_path + str(times) + "/"
                mkdir(model_save_path_times, clear=True)
                main(flag, use_list, sample_path, model_save_path_times, logfile, times)
        
        remove_item = eval_epoch(flag, inputs_list, epoch, sample_path)
        remove_list.append(remove_item)
        
        if remove_item == 0:
            break