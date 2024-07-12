import json
import os.path
import sys

import dill
import torch
import argparse

import yaml
import sys
import time
sys.path.append("/home/xiaopei/LogCAE/")

import utils.utils


from models.LogCAE import LogCAE

from demo.data_process_demo import getDataLoader
from utils import LossHistory

"""
得到数据后，完成离线训练部分：
对自编码器用MSE进行训练，训练后得到AE
"""
import logging
import time
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from sklearn.model_selection import train_test_split
from torch import optim, nn
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from tqdm import tqdm

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from models.AutoEncoder import AE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_f1 = 0
best_th = -1
utils.utils.set_seed()
def get_loss_threshold(y_preds, anomaly_ratio):
    # train_y_preds = torch.cat(y_preds, dim=0)
    # reconstruction_errors = train_y_preds
    # train_y_preds = y_preds.cpu().detach().numpy()

    # 使用 .numpy() 方法将新的张量转换为 NumPy 数组
    # y_preds = torch.stack(y_preds).cpu().detach().numpy()
    y_preds = np.array(y_preds)
    loss_thred = np.percentile(y_preds, 100 - anomaly_ratio * 100)

    return loss_thred



def get_loss_threshold_std(y_preds, n_std):
    # 使用 .numpy() 方法将新的张量转换为 NumPy 数组
    y_preds = torch.stack(y_preds).cpu().detach().numpy()
    loss_thred = y_preds.mean() + y_preds.std() * n_std

    return loss_thred

# 获取训练集样本的代表向量
def get_representative_vector(model, dataloader):
    vectors = []
    model.eval()
    with torch.no_grad():
        for inputs,targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            res_dict = model(inputs)
            vectors.append(res_dict["encoded"])
    vectors = torch.cat(vectors, dim=0)
    representative_vector = torch.mean(vectors, dim=0)
    distances = torch.norm(vectors - representative_vector, dim=(1, 2))

    # 获得每个向量的距离
    return representative_vector,distances

# 日志异常检测

def detect_anomalies(model, representative_vector, dataloader, threshold):
    """
    根据与特征向量的距离来进行异常检测
    :param model:
    :param representative_vector:
    :param dataloader:
    :param threshold:
    :return:
    """
    preds = []
    labels = []
    with torch.no_grad():
        for inputs,label in dataloader:
            inputs,label = inputs.to(device),label.to(device)
            res_dict = model(inputs)
            encodeds = res_dict["encoded"]
        y_pred = torch.norm(encodeds - representative_vector, dim=(1,2))
        # anomalies = distances > threshold
        preds.extend(y_pred.tolist())  # 移动 y_pred 到 cpu
        labels.extend(label.cpu())

    preds = np.array(preds)
    pred = (preds > threshold).astype(int)
    y = [int(label > 0) for label in labels]
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    # 计算混淆矩阵的值
    for i in range(len(pred)):
        if pred[i] == 1 and y[i] == 1:
            TP += 1
        elif pred[i] == 1 and y[i] == 0:
            FP += 1
        elif pred[i] == 0 and y[i] == 1:
            FN += 1
        elif pred[i] == 0 and y[i] == 0:
            TN += 1

    # 计算评价指标
    eval_results = {
        "f1": f1_score(y, pred),
        "rc": recall_score(y, pred),
        "pc": precision_score(y, pred),
        "acc": accuracy_score(y, pred),
    }
    CM = {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN
    }

    logging.info({k: f"{v:.4f}" for k, v in eval_results.items()})
    logging.info({k: f"{v:.4f}" for k, v in CM.items()})


    return eval_results


def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0.0
    y_preds = []
    epoch_time_start = time.time()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        res_dict = model(inputs)
        loss = torch.mean(res_dict["loss"])
        y_pred = res_dict["y_pred"]
        loss.backward()
        optimizer.step()

        y_preds.extend(y_pred.tolist())
        total_loss += loss.item()
        loss_history.append_loss(loss.item())


    epoch_loss = total_loss / len(train_loader)
    # loss_history.append_loss(epoch_loss)
    # epoch_time_elapsed = time.time() - epoch_time_start

    return epoch_loss, y_preds


def test_model(model, loss_thred, test_loader):
    # 初始化计数器
    TP = 0
    FP = 0
    FN = 0
    TN = 0

    with torch.no_grad():
        preds = []
        labels = []
        for seq, label in test_loader:
            seq, label = seq.to(device), label.to(device)
            return_dict = model(seq)
            y_pred = return_dict["y_pred"]
            preds.extend(y_pred.cpu())  # 移动 y_pred 到 cpu
            labels.extend(label.cpu())

        preds = np.array(preds)
        pred = (preds > loss_thred).astype(int)
        y = [int(label > 0) for label in labels]

        # 计算混淆矩阵的值
        for i in range(len(pred)):
            if pred[i] == 1 and y[i] == 1:
                TP += 1
            elif pred[i] == 1 and y[i] == 0:
                FP += 1
            elif pred[i] == 0 and y[i] == 1:
                FN += 1
            elif pred[i] == 0 and y[i] == 0:
                TN += 1

        # 计算评价指标
        eval_results = {
            "f1": f1_score(y, pred),
            "rc": recall_score(y, pred),
            "pc": precision_score(y, pred),
            "acc": accuracy_score(y, pred),
        }
        CM = {
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN
        }



        return preds,eval_results,CM

def test_th(model,y_preds,test_dataloader,mode="val",thresholds=[0.1,0.2,0.3,0.4,0.5]):

    flag = False
    for i in thresholds:
        th_5 = get_loss_threshold(y_preds, i)
        print(f"test dataloader({th_5},{i}):")
        preds,eval_results,CM = test_model(model, th_5, test_dataloader)
        if mode == "val" and eval_results["f1"] > best_f1:
            best_th = th_5
            best_f1 =  eval_results["f1"]
            logging.info({k: f"{v:.4f}" for k, v in eval_results.items()})
            logging.info({k: f"{v:.4f}" for k, v in CM.items()})
            flag = True


    return flag



def test_rep_vector(model,dataloader,representative_vector,distances,thresholds=np.arange(0, 1.1, 0.1)):
    best_score = 0
    best_threshold_precent = None
    for i in thresholds:
        distance_threshold = get_loss_threshold(distances,i)
        print(f"test_rep_vector({distance_threshold:.5f},{i}):")
        eval_results = detect_anomalies(model,representative_vector,dataloader,threshold=distance_threshold)

        if eval_results['f1'] > best_score:
            best_score = eval_results['f1']
            best_threshold_precent = i
    return best_score,best_threshold_precent

def offline_test(config,dataloader):

    train_dataloader,test_dataloader,al_dataloader = dataloader
    input_dim = config["dataset"]["emb_dim"]
    rnn_hidden_dim = config["Emb2Rep"]["rnn_hidden_dim"]
    rep_dim = config["Emb2Rep"]["rep_dim"]
    ae_hidden = config["AE"]["hidden_dims"]
    num_heads = config["Emb2Rep"]["num_heads"]
    # model = LogCAE(input_dim=input_dim,rnn_hidden_dim=rnn_hidden_dim,rep_dim=rep_dim,
    #                ae_hidden=ae_hidden,num_heads=num_heads)

    model = LogCAE(input_dim=input_dim, rnn_hidden_dim=rnn_hidden_dim, rep_dim=rep_dim,
                   ae_hidden=ae_hidden, num_heads=num_heads)
    # model = AE(input_dim=768, hidden_dims=config["AE"]["hidden_dims"])

    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"torch.cuda.device_count():{torch.cuda.device_count()}")
        model = nn.DataParallel(model)
    print(model)
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint)

    print("********Test**********")
    preds, eval_results, CM = test_model(model, best_th, test_dataloader)
    logging.info({k: f"{v:.4f}" for k, v in eval_results.items()})
    logging.info({k: f"{v:.4f}" for k, v in CM.items()})

    print("********Val**********")
    preds, eval_results, CM = test_model(model, best_th, al_dataloader)
    logging.info({k: f"{v:.4f}" for k, v in eval_results.items()})
    logging.info({k: f"{v:.4f}" for k, v in CM.items()})



# 获得代表向量
        # representative_vector,distances = get_representative_vector(model,train_dataloader)
        # distances = distances.tolist()

        # distance_threshold = get_loss_threshold(distances,config["anomaly_detection"]["ratio"])
        # detect_anomalies(model,representative_vector,test_dataloader,threshold=distance_threshold)
        # loss_threshold = get_loss_threshold(y_preds, config["anomaly_detection"]["ratio"])
        # loss_threshold2 = get_loss_threshold(y_preds, config["contrastive"]["ratio"])
        # test_th(model,y_preds,test_dataloader)

    #     thresholds = np.arange(0, 1.1, 0.1)
    #     # score,th_pre = test_rep_vector(model,test_dataloader,representative_vector,distances,thresholds)
    #
    #     print("*****Val_th*******")
    #     if flag:
    #         torch.save(model.state_dict(), save_path)
    #     print("*****test_th*******")
    #
    #     test_th(model,y_preds,test_dataloader,thresholds)
    #     logging.info(
    #         "Epoch {}/{}, training loss: {:.5f}".format(epoch, epochs, train_loss)
    #     )
    #     print("")
    # print("**********train end ************")
    #
    # return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AE")
    # parser.add_argument("--config", type=str, default="ae_bgl.yaml")  # ae_thunderbird
    # parser.add_argument("--config", type=str, default="zoo")  # ae_thunderbird
    parser.add_argument("--config", type=str, default="thu")  # ae_thunderbird
    args = parser.parse_args()
    # path = "config/" +

    path = f'../config/logcae_{args.config}.yaml'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # logging.basicConfig(level=logging.INFO)

    # with open(path, "r", encoding='utf-8-sig') as f:
    #     config = yaml.safe_load(f)

    result_path = f"../result/model_save/{args.config}/"
    # if not os.path.exists(result_path):
    #     os.makedirs(result_path)
    with open(result_path+"/reuslt.json",'rb')  as file:
        config = json.load(file)
    print(config)
    best_th =  config["best_result"]["threshold"]

    save_path =os.path.join(result_path,"model.pth")
    # dataloader = getDataLoader(config)
    # with open(os.path.join(result_path,"Dataloader.pkl"), 'wb') as f:
    #     dill.dump(dataloader,f)
    with open(os.path.join(result_path,"Dataloader.pkl"), 'rb') as f:
        dataloader = dill.load(f)
    model = offline_test(config,dataloader)

    # 保存



    # checkpoint = torch.load(save_path)
    # model.load_state_dict(checkpoint)

