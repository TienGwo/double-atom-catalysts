# -*- coding: utf-8 -*-

import sys
import torch
import numpy as np
import random

from sklearn.metrics import r2_score
from TopoCata.utils import print_all_info, print_monitor
from TopoCata.data.data_loader import build_dataset
from TopoCata.model.model_graph import TopoCata


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
    
class training:
    def __init__(self, epoch_num, lr, lr_decrease, step_size, decrease_rate):
        self.epoch_num = epoch_num
        self.lr = lr
        self.lr_decrease = lr_decrease
        self.step_size = step_size
        self.decrease_rate = decrease_rate
        
    
    def training_param(self, model):
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.MSE = torch.nn.MSELoss(reduction="mean")
        self.MAE = torch.nn.L1Loss(reduction="mean")
        
        if self.lr_decrease:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=self.optimizer,
                step_size=self.step_size,
                gamma=self.decrease_rate
                )
            
        
    def train(self, model, dataset):
        model.train()
        for samples in dataset:  # Iterate over each mini-batch.
            inputs, labels = samples
            outputs = model(inputs)  # Perform a single forward pass.
            loss = self.MSE(outputs, labels)
            loss.backward()  # Derive gradients.
            self.optimizer.step()  # Update parameters based on gradients.
            self.optimizer.zero_grad()  # Clear gradients
        return model
        
        
    @torch.no_grad()
    def valid(self, model, dataset):
        model.eval()
        MSE_record, MAE_record, MAPE_record = 0, 0, 0
        ML_pred, DFT_calc = [], []
        for index, sample in enumerate(dataset):  # Iterate over each mini-batch.
            inputs, labels = sample
            outputs = model(inputs)  # Perform a single forward pass.
            MSE = self.MSE(outputs, labels)  
            MAE = self.MAE(outputs, labels)  # Compute the loss solely based on the training nodes.
            MAPE = torch.mean(torch.abs((outputs - labels) / labels))
            
            for MD_data, DFT_data in zip(outputs, labels):
                ML_pred.append(MD_data.item())
                DFT_calc.append(DFT_data.item())
            
            MSE_record += MSE.item()
            MAE_record += MAE.item()
            MAPE_record += MAPE.item()
        
        MSE_record /= index + 1
        MAE_record /= index + 1
        MAPE_record /= index + 1
        R2_record = r2_score(y_true=DFT_calc, y_pred=ML_pred)
        
        return MSE_record, MAE_record, MAPE_record, R2_record


    def main(self, sample_path, model_save_path, flag, batch_size):
        
        logfile = flag + "_training_log.txt"
        print_all_info(
            sample_path=sample_path, 
            filename=logfile,
            lr=self.lr
            )
        
        # define the neural network and dataset
        model = TopoCata()
        dataset_train, dataset_valid = build_dataset(sample_path, flag, batch_size)
        
        # load the training parameters
        self.training_param(model)
        MSE_record = np.zeros([self.epoch_num, 2], dtype=np.float32)
        MAE_record = np.zeros_like(MSE_record)
        MAPE_record = np.zeros_like(MSE_record)
        R2_record = np.zeros_like(MSE_record)
        
        # start trainings
        R2_flag = 0
        for epoch in range(self.epoch_num):
            model = self.train(model, dataset_train) # training
            MSE_train, MAE_train, MAPE_train, R2_train = self.valid(model, dataset_train) # validating in training set
            MSE_valid, MAE_valid, MAPE_valid, R2_valid = self.valid(model, dataset_valid) # validating in testing set
            
            if self.lr_decrease: # decrease of learing rate
                self.scheduler.step()
            
            # recording
            MSE_record[epoch, 0] = MSE_train
            MSE_record[epoch, 1] = MSE_valid
            MAE_record[epoch, 0] = MAE_train
            MAE_record[epoch, 1] = MAE_valid
            MAPE_record[epoch, 0] = MAPE_train
            MAPE_record[epoch, 1] = MAPE_valid
            R2_record[epoch, 0] = R2_train
            R2_record[epoch, 1] = R2_valid
            
            
            if R2_valid > R2_flag and R2_valid < R2_train:
                
                R2_flag = R2_valid
                
                # model saving
                torch.save(
                    obj = model.state_dict(),
                    f = model_save_path + flag + "/model.pkl"
                    )
                
                lr = self.optimizer.param_groups[0]["lr"]
                print_monitor(
                    epoch, lr, 
                    MSE_train, MSE_valid, 
                    MAE_train, MAE_valid, 
                    MAPE_train, MAPE_valid, 
                    R2_train, R2_valid,
                    logfile
                    )
        
        # saving the record
        np.save(model_save_path + flag + "/MSE_record.npy", MSE_record)
        np.save(model_save_path + flag + "/MAE_record.npy", MAE_record)
        np.save(model_save_path + flag + "/RD_record.npy", MAPE_record)
        np.save(model_save_path + flag + "/R2_record.npy", R2_record)
            
            
if __name__ == "__main__":
    
    setup_seed(seed=1024)
    dataset_train, dataset_valid = build_dataset(
        sample_path=str(sys.argv[6]),
        flag=str(sys.argv[8]),
        batch_size=int(sys.argv[9])
        )
        
    R2_flag = 0
    for _ in range(100):
        
        container = training(
            epoch_num=int(sys.argv[1]),
            lr=float(sys.argv[2]),
            lr_decrease=bool(sys.argv[3]),
            step_size=float(sys.argv[4]),
            decrease_rate=float(sys.argv[5])
            )
        
        container.main(
            sample_path=str(sys.argv[6]),
            model_save_path=str(sys.argv[7]),
            flag=str(sys.argv[8]),
            batch_size=int(sys.argv[9])
            )
        
        model = TopoCata()
        model.load_state_dict(
            torch.load("./model_save/" + str(sys.argv[8]) + "/model.pkl")
            )
        model.eval()
        
        _, _, _, R2_train = container.valid(model, dataset_train) # validating in training set
        _, _, _, R2_valid = container.valid(model, dataset_valid) # validating in testing set
        
        if R2_valid >= R2_flag:
            R2_flag = R2_valid
            
            torch.save(
                obj = model.state_dict(),
                f = "./model_save/" + str(sys.argv[8]) + "/model_opt.pkl"
                )
            
            print("R2 value of train/valid:", "%.3f" % R2_train, "===", "%.3f" % R2_valid)