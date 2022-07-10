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
from Source.signals_processing import check_signal
from Source.gene_wam import WAM
from Source.gene_bayes import BN
from Source.gene_svm import SVM

# 加载donor site分类器
du_len, dd_len = 3, 6
donor_len = du_len + dd_len
model_dir = f"./Models/Bayes/donor/u3d6_u15d2/"
donor_clf = BN(du_len, dd_len, model_dir, "donor", "hillclimb")
donor_thr = 3.02
donor_clf.load_params()
# 加载acceptor site分类器
au_len, ad_len = 25, 9
acceptor_len = au_len + ad_len
model_dir = f"./Models/Bayes/acceptor/u9d15_u25d9/"
acceptor_clf = BN(au_len, ad_len, model_dir, "acceptor", "hillclimb")
acceptor_thr = 3.87
acceptor_clf.load_params()

# 选择文件
# dir_path = "./Data_files/raw_data/Testing Set/"
# file_name = os.listdir(dir_path)[0]
# with open(dir_path + file_name, 'r') as f:
#     content = f.readlines()
# 输入序列
content = input("请输入基因序列：").split()

whole_seq = "".join([row.strip() for row in content if not row.startswith('>')])

# 信号提取
signals = []
locs = []
for i in range(len(whole_seq) - donor_len + 1):
    slide = whole_seq[i: i + donor_len].lower()
    if slide[du_len: du_len + 2] == "gt" and check_signal(slide, donor_len):
        signals.append(slide)
        locs.append(i)

# 分类预测
print("Donor splicing site:")
labels = donor_clf.predict(signals, donor_thr)
for i in range(labels.shape[0]):
    if labels[i] == 1:
        print(locs[i] + du_len)

# 信号提取
signals = []
locs = []
for i in range(len(whole_seq) - acceptor_len + 1):
    slide = whole_seq[i: i + acceptor_len].lower()
    if slide[au_len - 2: au_len] == "ag" and check_signal(slide, acceptor_len):
        signals.append(slide)
        locs.append(i)

# 分类预测
print("Acceptor splicing site:")
labels = acceptor_clf.predict(signals, acceptor_thr)
for i in range(labels.shape[0]):
    if labels[i] == 1:
        print(locs[i] + au_len + 1)
