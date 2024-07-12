"""
观察分布：
阈值，正常样本（c)，异常样本（c)
绘制正常样本的分布曲线
绘制异常样本的分布曲线（
主动学习：
在al_dat

模型再主动学习集上进行推理得到结果，
选出在阈值附近的样本（百分之多少样本）

制作新的数据集

"""
import logging


def select_fuzzy_sample_according_distance(model,rep_v,distance_th,aldataloder,num_precent=0.1):
    logging.info("------Activet learning start------")
    logging.info(f"fuzzy sample loss ({loss_thred-fuzzy_th},{loss_thred+fuzzy_th})")



