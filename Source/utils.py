# -*- coding: utf-8 -*-
# @Time    : 2022/7/10 1:28
# @Author  : Terence Tan
# @Email   : 2228254095@qq.com
# @FileName: utils.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


def make_labels(pos_signals, neg_signals):
    pos_labels = [1 for _ in range(len(pos_signals))]
    neg_labels = [0 for _ in range(len(neg_signals))]
    labels = pos_labels + neg_labels

    return labels


def plot_roc(fpr, tpr, roc_auc, legend, title):
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.plot(fpr, tpr, 'b-', label=f"{legend}(AUC={roc_auc:.3f})", linewidth=1)
    plt.legend(loc="best")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"{title} ROC Curve")


def plot_pr(recall, precision, pr_auc, legend, title):
    plt.plot([0, 1], [1, 0], 'k--', linewidth=1)
    plt.plot(recall, precision, 'b-', label=f"{legend}(AUC={pr_auc:.3f})", linewidth=1)
    plt.legend(loc="best")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title} PR Curve")


def find_thr(labels, scores):
    """
    寻找使得f1-score最大的阈值
    """
    precision, recall, thresholds = metrics.precision_recall_curve(labels, scores)
    max_f1score = 0
    ulti_thr = 0.5
    thrs = np.arange(min(thresholds), max(thresholds), 0.05)

    for thr in thrs:
        labels_pred = np.zeros(scores.shape[0], dtype=int)
        labels_pred[scores >= thr] = 1
        f1score = metrics.f1_score(labels, labels_pred)
        if f1score > max_f1score:
            max_f1score = f1score
            ulti_thr = thr

    print(f"threshold: {ulti_thr}, max f1-score: {max_f1score}")
