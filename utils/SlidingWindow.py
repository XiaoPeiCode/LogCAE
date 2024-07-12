import os.path

import numpy as np
import pandas as pd
from tqdm import tqdm
from gensim.models import Word2Vec
from nltk.tokenize import RegexpTokenizer
import warnings
from .BART import bart_encode
import gc
warnings.filterwarnings("ignore")

def word2vec_train(lst, emb_dim = 150, seed = 42):
    """
    train a word2vec mode
    args: lst(list of string): sentences
          emb_dim(int): word2vec embedding dimensions
          seed(int): seed for word2vec
    return: word2vec model
    """
    tokenizer = RegexpTokenizer(r'\w+')
    sentences = []
    for i in lst:
        sentences.append([x.lower() for x in tokenizer.tokenize(str(i))])
    w2v = Word2Vec(sentences, vector_size=emb_dim, min_count=1, seed=seed)
    return w2v

def get_sentence_emb(sentence, w2v):
    """
    get a sentence embedding vector
    *automatic initial random value to the new word
    args: sentence(string): sentence of log message
          w2v: word2vec model
    return: sen_emb(list of int): vector for the sentence
    """
    tokenizer = RegexpTokenizer(r'\w+')
    lst = []
    tokens = [x.lower() for x in tokenizer.tokenize(str(sentence))]
    if tokens == []:
        tokens.append('EmptyParametersTokens')
    for i in range(len(tokens)):
        words = w2v.wv.index_to_key
        if tokens[i] in words:
            lst.append(w2v.wv[tokens[i]])
        else:
            w2v.build_vocab([[tokens[i]]], update = True)
            w2v.train([tokens[i]], epochs=1, total_examples=len([tokens[i]]))
            lst.append(w2v.wv[tokens[i]])
    drop = 1
    if len(np.array(lst).shape) >= 2:
        sen_emb = np.mean(np.array(lst), axis=0)
        if len(np.array(lst)) >= 5:
            drop = 0
    else:
        sen_emb = np.array(lst)
    return list(sen_emb), drop

def word2emb(df_source, df_target, emb_dim, all_templates):
    w2v = word2vec_train(all_templates, emb_dim=emb_dim)
    print('Processing words in the source dataset')
    dic = {}
    lst_temp = list(set(df_source.EventTemplate.values))
    for i in tqdm(range(len(lst_temp))):
        (temp_val, drop) = get_sentence_emb([lst_temp[i]], w2v)
        dic[lst_temp[i]] = (temp_val, drop)
    lst_emb = []
    lst_drop = []
    for i in tqdm(range(len(df_source))):
        lst_emb.append(dic[df_source.EventTemplate.loc[i]][0])
        lst_drop.append(dic[df_source.EventTemplate.loc[i]][1])
    df_source['Embedding'] = lst_emb
    df_source['drop'] = lst_drop
    print('Processing words in the target dataset')
    dic = {}
    lst_temp = list(set(df_target.EventTemplate.values))
    for i in tqdm(range(len(lst_temp))):
        (temp_val, drop) = get_sentence_emb([lst_temp[i]], w2v)
        dic[lst_temp[i]] = (temp_val, drop)
    lst_emb = []
    lst_drop = []
    for i in tqdm(range(len(df_target))):
        lst_emb.append(dic[df_target.EventTemplate.loc[i]][0])
        lst_drop.append(dic[df_target.EventTemplate.loc[i]][1])
    df_target['Embedding'] = lst_emb
    df_target['drop'] = lst_drop

    df_source = df_source.loc[df_source['drop'] == 0]
    df_target = df_target.loc[df_target['drop'] == 0]

    print(f'Source length after drop none word logs: {len(df_source)}')
    print(f'Target length after drop none word logs: {len(df_target)}')

    return df_source, df_target, w2v

def bart_emb(df_source, df_target):
    # index1 = df_source[df_source['Label'] != '-'].index[0]
    # index2 = df_target[df_target['Label'] != '-'].index[1]
    # df_source = df_source.iloc[index1:index1+4000]
    # df_target = df_target.iloc[index2:index2+4000]

    n_source = len(df_source)
    source_corpus = df_source.EventTemplate.values
    target_corpus = df_target.EventTemplate.values


    corpus = np.concatenate([source_corpus,target_corpus])

    X = bart_encode(corpus)

    source_embs = X[:n_source]
    target_embs = X[n_source:]

    df_source['Embedding'] = [emb for emb in source_embs]
    df_target['Embedding'] = [emb for emb in target_embs]

    return df_source, df_target


def sliding_window(df, window_size = 20, step_size = 4):
    df["Label"] = df["Label"].apply(lambda x: int(x != '-'))
    df = df[["Label", "Embedding"]]
    log_emb_seqs = []
    log_labels = []
    log_size = df.shape[0]
    label_data = df.iloc[:, 0]
    emb_data = []
    for i in df.iloc[:, 1].values:
        emb_data.append(np.array(i))
    for index in tqdm(range(log_size-window_size)):
        if len(log_emb_seqs) > 1000000:
            break
        log_emb_seqs.append(np.array(emb_data[index:index + window_size]))
        log_labels.append(max(label_data[index:index + window_size]))
        index += step_size
    log_emb_seqs = np.array(log_emb_seqs)
    log_labels = np.array(log_labels)
    return log_emb_seqs,log_labels

def sliding_window_HDFS(df, window_size = 20, step_size = 4):
    log_emb_seqs = []
    log_labels = []
    df = df[["Label", "Embedding", "BlockId"]]
    dict = {}
    label = {}
    for idx, row in df.iterrows():
        if row['BlockId'] not in dict:
            dict[row['BlockId']] = []
            label[row['BlockId']] = row['Label']
        dict[row['BlockId']].append(row['Embedding'])
    for blockid in tqdm(dict):
        if len(log_emb_seqs) > 400000:
            break
        log_size = len(dict[blockid])
        index = 0
        while index <= log_size - window_size:
            log_emb_seqs.append(np.array(dict[blockid][index:index + window_size]))
            log_labels.append(label[blockid])
            index += step_size
    log_emb_seqs = np.array(log_emb_seqs)
    log_labels = np.array(log_labels)
    return log_emb_seqs, log_labels

def sliding_window_Zookeeper(df, window_size = 20, step_size = 4):
    df["Label"] = df["Level"].apply(lambda x: int(x == 'ERROR'))
    df = df[["Label", "Embedding"]]
    log_emb_seqs = []
    log_labels = []
    log_size = df.shape[0]
    label_data = df.iloc[:, 0]
    emb_data = []
    for i in df.iloc[:, 1].values:
        emb_data.append(np.array(i))
    for index in tqdm(range(log_size-window_size)):
        if len(log_emb_seqs) > 400000:
            break
        log_emb_seqs.append(np.array(emb_data[index:index + window_size]))
        log_labels.append(max(label_data[index:index + window_size]))
        index += step_size
    log_emb_seqs = np.array(log_emb_seqs)
    log_labels = np.array(log_labels)
    return log_emb_seqs,log_labels

def get_datasets_word2vec(df_source, df_target, config):
    func_dict = {
        'BGL': sliding_window,
        'HDFS': sliding_window_HDFS,
        'Thunderbird': sliding_window,
        'Zookeeper': sliding_window_Zookeeper
    }
    # Get source data preprocessed
    window_size = config["window_size"]
    step_size = config["step_size"]

    source = config["source_dataset_name"]
    target = config["target_dataset_name"]
    emb_dim = config["emb_dim"]

    # save path
    s_log_seqs_path = f'./dataset/{source}/{source}_to_{target}_log_seqs.npy'
    s_log_labels_path = f'./dataset/{source}/{source}_to_{target}_log_labels.npy'
    t_log_seqs_path = f'./dataset/{target}/{source}_to_{target}_log_seqs.npy'
    t_log_labels_path = f'./dataset/{target}/{source}_to_{target}_log_labels.npy'
    
    # templates path
    s_log_templates_path = './{}/{}/{}.log_templates.csv'.format(
        config['dir'], config['source_dataset_name'],
        config['source_dataset_name'])
    t_log_templates_path = './{}/{}/{}.log_templates.csv'.format(
        config['dir'], config['source_dataset_name'],
        config['source_dataset_name'])
    df_source_templates = pd.read_csv(s_log_templates_path)
    df_target_templates = pd.read_csv(t_log_templates_path)
    all_templates = np.concatenate((df_source_templates.EventTemplate.values,df_target_templates.EventTemplate.values))



    df_source, df_target, w2v = word2emb(df_source, df_target, emb_dim, all_templates)


    print(f'preprocessing for the dataset: {source} and {target}')
    s_log_seqs,s_log_labels = func_dict[source](df_source, window_size, step_size)
    np.save(s_log_seqs_path, s_log_seqs)
    np.save(s_log_labels_path, s_log_labels)
    del df_source
    del s_log_seqs
    del s_log_labels
    t_log_seqs,t_log_labels = func_dict[target](df_target, window_size, step_size)
    del df_target
    np.save(t_log_seqs_path, t_log_seqs)
    np.save(t_log_labels_path, t_log_labels)

def get_datasets(df_source, df_target, options, val_date="2005.11.15"):
    # Get source data preprocessed
    window_size = options["window_size"]
    step_size = options["step_size"]
    source = options["source_dataset_name"]
    target = options["target_dataset_name"]
    train_size_s = options["train_size_s"]
    train_size_t = options["train_size_t"]
    emb_dim = options["emb_dim"]
    times =  int(train_size_s/train_size_t) - 1

    df_source, df_target, w2v = word2emb(df_source, df_target, train_size_s, train_size_t, step_size, emb_dim)

    print(f'Start preprocessing for the source: {source} dataset')
    window_df = sliding_window(df_source, window_size, step_size, 0, val_date)
    r_s_val_df = window_df[window_df['val'] == 1]
    window_df = window_df[window_df['val'] == 0]

    # Training normal data
    df_normal = window_df[window_df["Label"] == 0]

    # shuffle normal data
    df_normal = df_normal.sample(frac=1, random_state=42).reset_index(drop=True)
    train_len = train_size_s

    train_normal_s = df_normal[:train_len]
    print("Source training size {}".format(len(train_normal_s)))

    # Test normal data
    test_normal_s = df_normal[train_len:]
    print("Source test normal size {}".format(len(test_normal_s)))

    # Testing abnormal data
    test_abnormal_s = window_df[window_df["Label"] == 1]
    print('Source test abnormal size {}'.format(len(test_abnormal_s)))

    print('------------------------------------------')
    print(f'Start preprocessing for the target: {target} dataset')
    # Get target data preprocessed
    window_df = sliding_window(df_target, window_size, step_size, 1, val_date)
    r_t_val_df = window_df[window_df['val'] == 1]
    window_df = window_df[window_df['val'] == 0]

    # Training normal data
    df_normal = window_df[window_df["Label"] == 0]
    # shuffle normal data
    df_normal = df_normal.sample(frac=1, random_state=42).reset_index(drop=True)
    train_len = train_size_t

    train_normal_t = df_normal[:train_len]
    print("Target training size {}".format(len(train_normal_t)))
    temp = train_normal_t[:]
    for _ in range(times):
        train_normal_t = pd.concat([train_normal_t, temp])

    # Testing normal data
    test_normal_t = df_normal[train_len:]
    print("Target test normal size {}".format(len(test_normal_t)))

    # Testing abnormal data
    test_abnormal_t = window_df[window_df["Label"] == 1]
    print('Target test abnormal size {}'.format(len(test_abnormal_t)))

    return train_normal_s, test_normal_s, test_abnormal_s, r_s_val_df, \
           train_normal_t, test_normal_t, test_abnormal_t, r_t_val_df, w2v

def get_datasets_bart(df_source, df_target, config):
    func_dict = {
        'BGL': sliding_window,
        'HDFS': sliding_window_HDFS,
        'Thunderbird': sliding_window,
        'Zookeeper': sliding_window_Zookeeper
    }
    # Get source data preprocessed
    window_size = config["window_size"]
    step_size = config["step_size"]

    source = config["source_dataset_name"]
    target = config["target_dataset_name"]
    emb_dim = config["emb_dim"]

    # save path
    s_log_seqs_path = f'./dataset/{source}/{source}_to_{target}_log_seqs.npy'
    s_log_labels_path = f'./dataset/{source}/{source}_to_{target}_log_labels.npy'
    t_log_seqs_path = f'./dataset/{target}/{source}_to_{target}_log_seqs.npy'
    t_log_labels_path = f'./dataset/{target}/{source}_to_{target}_log_labels.npy'

    # templates path
    s_log_templates_path = './{}/{}/{}.log_templates.csv'.format(
        config['dir'], config['source_dataset_name'],
        config['source_dataset_name'])
    t_log_templates_path = './{}/{}/{}.log_templates.csv'.format(
        config['dir'], config['source_dataset_name'],
        config['source_dataset_name'])
    df_source_templates = pd.read_csv(s_log_templates_path)
    df_target_templates = pd.read_csv(t_log_templates_path)
    all_templates = np.concatenate((df_source_templates.EventTemplate.values, df_target_templates.EventTemplate.values))

    df_source, df_target = bart_emb(df_source, df_target)

    print(f'preprocessing for the dataset: {source} and {target}')
    s_log_seqs, s_log_labels = func_dict[source](df_source, window_size, step_size)
    np.save(s_log_seqs_path, s_log_seqs)
    np.save(s_log_labels_path, s_log_labels)
    del df_source
    del s_log_seqs
    del s_log_labels
    t_log_seqs, t_log_labels = func_dict[target](df_target, window_size, step_size)
    del df_target
    np.save(t_log_seqs_path, t_log_seqs)
    np.save(t_log_labels_path, t_log_labels)

def get_datasets_bart_one(df_source, config):
    func_dict = {
        'BGL': sliding_window,
        'HDFS': sliding_window_HDFS,
        'Thunderbird': sliding_window,
        'Zookeeper': sliding_window_Zookeeper
    }
    # Get source data preprocessed
    window_size = config["window_size"]
    step_size = config["step_size"]

    source = config["dataset_name"]
    emb_dim = config["emb_dim"]

    # save path
    s_log_seqs_path = f'./dataset/{source}/{source}_log_seqs.npy'
    s_log_labels_path = f'./dataset/{source}/{source}_log_labels.npy'


    # templates path
    s_log_templates_path = './{}/{}/{}.log_templates.csv'.format(
        config['dir'], config['dataset_name'],
        config['dataset_name'])

    # df_source_templates = pd.read_csv(s_log_templates_path)
    # all_templates = np.concatenate((df_source_templates.EventTemplate.values, df_target_templates.EventTemplate.values))
    # df_source, df_target = bart_emb(df_source, df_target)
    corpus = df_source.EventTemplate.values

    bart_encode_path = f'./dataset/{source}/{source}_bart_encode_{config["total_num"]}.npy'
    bart_encode_path = f'./dataset/{source}/{source}_bart_encode.npy'
    if not os.path.exists(bart_encode_path): #  or config["need_Bart"]
        print(f'bart encoding:')
        X = bart_encode(corpus)
        np.save(bart_encode_path, X)
    else:
        print(f'bart_encode_path exit')
        X = np.load(bart_encode_path)

    df_source['Embedding'] = [emb for emb in X]

    if config["total_num"] != "all":
        # df_source = df_source.head()
        start_index = config["start_index"]
        total_num = config["total_num"]
        print(f"start_index:{start_index},total_num:{total_num}")
        df_source = df_source.iloc[start_index:start_index + total_num]
    print(f'preprocessing for the dataset: {source}')
    s_log_seqs, s_log_labels = func_dict[source](df_source, window_size, step_size)
    np.save(s_log_seqs_path, s_log_seqs)
    np.save(s_log_labels_path, s_log_labels)
    del df_source
    del s_log_seqs
    del s_log_labels



