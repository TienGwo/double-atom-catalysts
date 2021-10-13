# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

class MyDataset(Dataset):
    def __init__(self, inputsfile, labelsfile, use_list):
        self.use_list = use_list
        self.inputs_train = np.load(inputsfile)
        self.labels_train = np.load(labelsfile)
    
    def __len__(self):
        return self.inputs_train.shape[0]
    
    def __getitem__(self, index):
        inputs = torch.from_numpy(self.inputs_train[index, self.use_list, :].T).float()
        labels = torch.from_numpy(self.labels_train[index, :]).float()
        return inputs, labels
    
        
def build_dataset(sample_path, flag, use_list):

    dataset_train = DataLoader(
        dataset=MyDataset(
            inputsfile=sample_path + "inputs_train_" + flag + ".npy",
            labelsfile=sample_path + "labels_train_" + flag + ".npy",
            use_list=use_list
            ), 
        batch_size=12, 
        shuffle=True
        )

    dataset_valid = DataLoader(
        dataset=MyDataset(
            inputsfile=sample_path + "inputs_valid_" + flag + ".npy",
            labelsfile=sample_path + "labels_valid_" + flag + ".npy",
            use_list=use_list
            ), 
        batch_size=22,
        shuffle=True
        )
    
    return dataset_train, dataset_valid
        
        

if __name__ == "__main__":
    
    dataset_train, dataset_valid = build_dataset(
        sample_path="./dataset/", 
        flag="BE", 
        batch_size=15,
        use_list=[_ for _ in range(8)]
        )
    
    for index, data in enumerate(dataset_train):
        inputs, labels = data
        
    print(inputs.shape)
    print(labels.shape)