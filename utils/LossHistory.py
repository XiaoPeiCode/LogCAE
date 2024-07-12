import os

import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn

class LossHistory():
    def __init__(self, log_dir):
        import datetime
        curr_time = datetime.datetime.now()
        time_str = datetime.datetime.strftime(curr_time, '%Y_%m_%d_%H_%M_%S')
        self.log_dir = log_dir
        self.time_str = time_str
        self.save_path = os.path.join(self.log_dir, "loss_" + str(self.time_str))
        self.losses = []


        os.makedirs(self.save_path)

    def append_loss(self, loss, val_loss=None):
        self.losses.append(loss)
        with open(os.path.join(self.save_path, "Iter_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        # plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15

            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle='--', linewidth=2,
                     label='smooth train loss')

        except:
            pass

        plt.grid(True)
        plt.xlabel('Iter')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.save_path, "Iter_loss_" + str(self.time_str) + ".png"))
        # plt.show()
        plt.cla()
        plt.close("all")

if __name__ == '__main__':
    pass
    # loss_history = LossHistory.LossHistory(save_path+"/logs_Finetue/")
    #
    # loss_history.append_loss(sum_loss / num_batches)