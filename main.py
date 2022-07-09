# -*- coding: utf-8 -*-
# @Time    : 2022/6/23 13:51
# @Author  : Terence Tan
# @Email   : 2228254095@qq.com
# @FileName: main.py
# @Software: PyCharm

import sys
sys.path.append("./Source")
import os
from sklearn import metrics
from Source.gene_wam import WAM
from Source.gene_bayes import BN
from Source.gene_svm import SVM

# 加载模型
up_len, down_len = 3, 6
signal_len = up_len + down_len
# model_dir = f"./Models/WAM/u3d6_u12d5/"
# donor_clf = WAM(up_len, down_len, model_dir)
# thr = 2.61
model_dir = f"./Models/Bayes/u3d6_u12d5/"
donor_clf = BN(up_len, down_len, model_dir)
thr = 3.02
# 加载参数
donor_clf.load_params()

# 提取基因序列
dir_path = "./Data_files/raw_data/Testing Set/"
file_name = os.listdir(dir_path)[2]
with open(dir_path + file_name, 'r') as f:
    content = f.readlines()
    whole_seq = "".join([row.strip() for row in content if not row.startswith('>')])

# 信号提取
signals = []
locs = []
for i in range(len(whole_seq) - signal_len + 1):
    slide = whole_seq[i: i + signal_len].lower()
    if slide[up_len: up_len + 2] == "gt":
        signals.append(slide)
        locs.append(i)

# 分类预测
labels = donor_clf.predict(signals, thr)
for i in range(labels.shape[0]):
    if labels[i] == 1:
        print(locs[i] + up_len)

