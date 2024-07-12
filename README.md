# LogCAE

# 架构设计

1. 数据预处理部分
2. 自编码器训练（阈值检测）
3. 主动学习模块
## 整个流程
(一)数据预处理 

BGL:
- 原始日志数据：BGL.log
- parsing后：BGL.log_structured.csv(解析的结构化信息)+templates.csv(日志模板)
- Bart encoder：对log_structured.csv中的EventTemplate.values进行bart编码
  - {source}_bart_encode_{config["total_num"]}.npy
- 获取日志序列，制作数据集：sliding_window
  - 得到
    s_log_seqs_path = f'./dataset/{source}/{source}_log_seqs.npy'
    s_log_labels_path = f'./dataset/{source}/{source}_log_labels.npy'
- 划分训练集、测试集、验证集





   1. 获得全部解析数据集log_structured.csv
       preprocessing.parsing(config['dataset_name'], config['dir'])
      - 下载数据-> 解析数据 -> bart预训练得到（bart encoder以及窗口截取后的数据log_seqs)
      - 
      > 这里不需要重新来过
   2. 制作训练数据集（log_seqs,log_labels) config["preprocess"] = True
      - 从log_structured.csv中截取config["dataset"]["total_num"]
      - 进行bart编码：bart_encode(corpus)
      - 窗口数据集制作：sliding_window
        - config["dataset"]["step_size"] :窗口滑动的步长
        - HDFS: 在每个BlockID中切分

