import json
import logging
import os

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import TensorDataset, DataLoader

from utils import preprocessing, SlidingWindow
from utils.utils import set_seed


def pre_process(config):
    """
    :param config:
    :return:
    """
    log_structured_path = './{}/{}/{}.log_structured.csv'.format(
        config['dir'], config['dataset_name'],
        config['dataset_name'])
    if not os.path.exists(log_structured_path):
        preprocessing.parsing(config['dataset_name'], config['dir'])

    df_source = pd.read_csv(log_structured_path)
    print(f'Reading source dataset: {config["dataset_name"]}; len(all dataset)={len(df_source)}')

    # if config["total_num"] != "all":
    #     # df_source = df_source.head()
    #     start_index = config["start_index"]
    #     total_num = config["total_num"]
    #     print(f"start_index:{start_index},total_num:{total_num}")
    #
    #     df_source = df_source.iloc[start_index:start_index + total_num]

    return SlidingWindow.get_datasets_bart_one(df_source, config)

def split_dataset_three(log_vectors, log_labels, train_ratio=0.7,test_ratio=0.5):
    # 过滤出标签为0和1的索引
    indices_label_0 = np.where(log_labels == 0)[0]
    indices_label_1 = np.where(log_labels == 1)[0]
    logging.info(f"total  len(label=0):{len(indices_label_0)}, len(label=1): {len(indices_label_1)}")

    # 将ratio%的normal log作为pre_dataset
    np.random.shuffle(indices_label_0)
    split_index = int(len(indices_label_0) * train_ratio)
    train_indices_label_0 = indices_label_0[:split_index]
    train_log_vectors = log_vectors[train_indices_label_0]
    train_log_labels = log_labels[train_indices_label_0]

    # 将剩下的normal划分为test和val
    left_indices_label_0 = indices_label_0[split_index:]
    left_index = int(len(left_indices_label_0) * test_ratio)
    test_indices_label_0 = left_indices_label_0[:left_index]

    ## tricks
    # 计算抽取的元素数量
    # num_elements_to_sample = int(len(test_indices_label_0) * 0.5)
    # 抽取随机元素
    # test_indices_label_0 = random.sample(test_indices_label_0.tolist(), num_elements_to_sample)
    ## tricks


    test_log_vectors = log_vectors[test_indices_label_0]
    test_log_labels = log_labels[test_indices_label_0]

    al_indices_label_0 = left_indices_label_0[left_index:]
    al_log_vectors = log_vectors[al_indices_label_0]
    al_log_labels = log_labels[al_indices_label_0]

    #  将abnormal log也添加到test_datatse,al_dataset
    np.random.shuffle(indices_label_1)
    split_index_1 = int(len(indices_label_1) * test_ratio)
    ## tricks
    test_indices_label_1 = indices_label_1[:split_index_1]
    # test_indices_label_1 = indices_label_1[:]
    ## tricks

    al_indices_label_1 = indices_label_1[split_index_1:]

    test_log_vectors_1 = log_vectors[test_indices_label_1]
    test_log_labels_1 = log_labels[test_indices_label_1]
    al_log_vectors_1 = log_vectors[al_indices_label_1]
    al_log_labels_1 = log_labels[al_indices_label_1]


    # 合并 （包含所有标签为1的样本和部分标签为0的样本）
    test_log_vectors = np.concatenate((log_vectors[test_indices_label_0], log_vectors[test_indices_label_1]))
    test_log_labels = np.concatenate((log_labels[test_indices_label_0], log_labels[test_indices_label_1]))
    print(f"test dataset len(normal)={len(test_indices_label_0)},len(abnormal)={len(test_indices_label_1)}")

    al_log_vectors = np.concatenate((log_vectors[al_indices_label_0], log_vectors[al_indices_label_1]))
    al_log_labels = np.concatenate((log_labels[al_indices_label_0], log_labels[al_indices_label_1]))
    print(f"al dataset len(normal)={len(al_indices_label_0)},len(abnormal)={len(al_indices_label_1)}")

    print(f"pre dataset len={len(train_indices_label_0)}")
    pre_dataset = TensorDataset(torch.tensor(train_log_vectors, dtype=torch.float), torch.tensor(train_log_labels))
    test_dataset = TensorDataset(torch.tensor(test_log_vectors, dtype=torch.float), torch.tensor(test_log_labels))
    al_dataset = TensorDataset(torch.tensor(al_log_vectors, dtype=torch.float), torch.tensor(al_log_labels))

    # # 制作对比学习数据集
    # test_abnormal_dataset = TensorDataset(torch.tensor(log_vectors[indices_label_1], dtype=torch.float), torch.tensor(log_labels[indices_label_1]))
    # paired_abnormal = PairedDataset(test_abnormal_dataset, pre_dataset, 1)

    return pre_dataset, test_dataset , al_dataset

def getDataLoader(config):
    seed = config['global']['random_seed']
    set_seed(seed)
    dataset_name = config['dataset']["dataset_name"]

    log_seqs_path = f'./dataset/{dataset_name}/{dataset_name}_log_seqs.npy'
    log_labels_path = f'./dataset/{dataset_name}/{dataset_name}_log_labels.npy'

    if not os.path.exists(log_labels_path):
        config['global']['need_preprocess'] = True

    if config['global']['need_preprocess']:
        pre_process(config['dataset'])

    # 获得全部的log的emb
    log_embs = np.load(log_seqs_path)
    log_labels = np.load(log_labels_path)

    # 从emb中拆分出训练集（normal)，测试集和主动学习集
    train_ratio = config["dataset"]["train_ratio"]
    test_ratio = config["dataset"]["test_ratio"]
    pre_dataset, test_dataset, al_dataset = split_dataset_three(log_embs, log_labels, train_ratio=train_ratio,test_ratio=test_ratio)
    batch_size = config["train"]["batch_size"]
    train_dataloader = DataLoader(pre_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    print(f"len(pre_dataset)={len(pre_dataset)}")
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    # al_dataloader =
    if len(al_dataset) > 0:
        al_dataloader = DataLoader(al_dataset, batch_size=batch_size,shuffle=True, pin_memory=True)

    return train_dataloader,test_dataloader,al_dataloader
if __name__ == '__main__':
    path = './config/logcae_bgl.yaml'
    with open(path, "r", encoding='utf-8-sig') as f:
        config = yaml.safe_load(f)
    print(json.dumps(config, indent=4))
    getDataLoader(config)
