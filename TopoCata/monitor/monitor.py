# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use("Agg")

import sys
import numpy as np
import matplotlib.pyplot as plt

import torch
from TopoCata.model.model_graph import TopoCata
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def relative_error(pred, real):
    return np.mean(np.abs((pred - real) / real))


class monitor:
    def __init__(self, sample_path, flag):
        self.sample_path = sample_path
        self.flag = flag
    
    def load_data(self):
        self.train_inputs = torch.from_numpy(
            np.load(self.sample_path + "inputs_train_" + self.flag + ".npy")
            ).float()

        self.valid_inputs = torch.from_numpy(
            np.load(self.sample_path + "inputs_valid_" + self.flag + ".npy")
            ).float()

        self.train_labels = np.load(self.sample_path + "labels_train_" + self.flag + ".npy")
        self.valid_labels = np.load(self.sample_path + "labels_valid_" + self.flag + ".npy")
    
    
    def load_model(self):
        self.model = TopoCata()
        self.model.load_state_dict(
            torch.load("./model_save/" + self.flag + "/model_opt.pkl")
            )
        
        
    @torch.no_grad()
    def prediction(self):
        self.load_data()
        self.load_model()
        self.model.eval()
        self.train_pred = self.model(self.train_inputs.transpose(1, 2)).detach().numpy()
        self.valid_pred = self.model(self.valid_inputs.transpose(1, 2)).detach().numpy()
    
    
    def plot_regression(self):
        self.prediction()
        range_max = max(np.max(self.train_pred), np.max(self.train_labels)) + 1.5
        range_min = min(np.min(self.valid_pred), np.min(self.valid_labels)) - 1.5

        fontsize = 15
        plt.figure(figsize=(6, 6))
        plt.scatter(self.train_pred, self.train_labels, s=100, c="g", ec="k", linewidth=1.5, alpha=0.6, zorder=1)
        plt.scatter(self.valid_pred, self.valid_labels, s=100, c="b", ec="k", linewidth=1.5, alpha=0.6, zorder=1)
        plt.plot([range_min, range_max], [range_min, range_max], c="k", ls="-.", lw=2, alpha=0.7, zorder=0)
        plt.xlabel("ML Predictions [eV]", fontsize=fontsize)
        plt.ylabel("DFT calculations [eV]", fontsize=fontsize)
        plt.title(self.flag, fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.savefig("./figs/" + self.flag + ".jpg", dpi=100, bbox_inches="tight")
        plt.close()
        
        np.savetxt("./logs/train_real_" + self.flag + ".txt", self.train_labels)
        np.savetxt("./logs/train_pred_" + self.flag + ".txt", self.train_pred)
        np.savetxt("./logs/valid_real_" + self.flag + ".txt", self.valid_labels)
        np.savetxt("./logs/valid_pred_" + self.flag + ".txt", self.valid_pred)
        
        print("\n" + "=" * 5, self.flag, "=" * 5, "\n")
        print("Train R2: ", "%.2f" % r2_score(y_true=self.train_labels, y_pred=self.train_pred))
        print("Valid R2: ", "%.2f" % r2_score(y_true=self.valid_labels, y_pred=self.valid_pred))
        print("Train RD: ", "%.2f" % (relative_error(self.train_labels, self.train_pred)  * 100) + " %")
        print("Valid RD: ", "%.2f" % (relative_error(self.valid_labels, self.valid_pred)  * 100) + " %")
        print("Train MAE:", "%.2f" % mean_absolute_error(self.train_labels, self.train_pred))
        print("Valid MAE:", "%.2f" % mean_absolute_error(self.valid_labels, self.valid_labels), "\n")
        
        
if __name__ == "__main__":
    
    container = monitor(
        sample_path=str(sys.argv[1]), 
        flag=str(sys.argv[2])
        )
    
    container.plot_regression()