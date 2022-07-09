# -*- coding: utf-8 -*-
# @Time    : 2022/4/12 10:40
# @Author  : Terence Tan
# @Email   : 2228254095@qq.com
# @FileName: gene_wam.py
# @Software: PyCharm

import os
import re
import datetime
import random
from functools import reduce
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from signals_processing import load_signals


class WAM:

    def __init__(self, signal_ulen, signal_dlen, model_dir):
        self.bases = {'a': 0, 'c': 1, 'g': 2, 't': 3}
        self.signal_ulen = signal_ulen
        self.signal_dlen = signal_dlen
        self.signal_len = signal_ulen + signal_dlen
        self.model_dir = model_dir

        # initialize wmm weights
        self.weight_matrix_init = np.zeros((self.signal_len, 4), dtype=np.float64)
        self.positive_weight_matrix = self.weight_matrix_init
        self.negative_weight_matrix = self.weight_matrix_init
        # initialize wam weights
        self.weight_array_init = np.zeros((self.signal_len - 1, 4, 4), dtype=np.float64)
        self.positive_weight_array = self.weight_array_init
        self.negative_weight_array = self.weight_array_init

    def fit(self, pos_signals, neg_signals):
        """
        训练模型
        """
        if len(pos_signals) > 0:
            print('Fitting model with positive data...')
            self.positive_weight_matrix, self.positive_weight_array = self._compute_weights(pos_signals)
        if len(neg_signals) > 0:
            print('Fitting model with negative data...')
            self.negative_weight_matrix, self.negative_weight_array = self._compute_weights(neg_signals)

        self._trans_weights_format()
        self._plot_weights()

        # save wmm and wam weights
        self._save_params()

    def _compute_weights(self, signals, psuedo_count=1e-6):
        """
        计算模型权重
        """
        # compute wmm weights
        weight_matrix = self.weight_matrix_init.copy()
        for signal in signals:
            for i in range(len(signal)):
                bx = self.bases[signal[i].lower()]
                weight_matrix[i, bx] += 1.0
        weight_matrix = weight_matrix / len(signals)
        weight_matrix[weight_matrix == 0] = psuedo_count

        # compute wam weights
        weight_array = self.weight_array_init.copy()
        for signal in signals:
            for i in range(1, len(signal)):
                bx = self.bases[signal[i - 1].lower()]
                by = self.bases[signal[i].lower()]
                weight_array[i - 1, bx, by] += 1.0
        weight_array = weight_array / len(signals)
        for i in range(weight_array.shape[0]):
            for x in range(weight_array.shape[1]):
                weight_array[i, x, :] = weight_array[i, x, :] / weight_matrix[i, x]
        weight_array[weight_array == 0] = psuedo_count

        return weight_matrix, weight_array

    def _base_arrays(self, arr_dim):
        """
        生成权重矩阵的列名
        """
        bases = list(map(str.upper, self.bases.keys()))
        arrays = reduce(lambda x, y: [i + j for i in x for j in y],
                        [bases] * arr_dim)

        return arrays

    def _row_names(self, arr_dim):
        """
        生成权重矩阵的行名
        """
        position = list(map(str, list(range(-self.signal_ulen, 0)) + list(range(1, self.signal_dlen + 1))))
        adj_pos = [f"({position[i-1]},{position[i]})" for i in range(1, len(position))]

        return position if arr_dim == 1 else adj_pos

    def _trans_weights_format(self):
        """
        将权重矩阵的数据格式由array转换为DataFrame
        """
        self.pwm = pd.DataFrame(data=self.positive_weight_matrix,
                                index=self._row_names(1),
                                columns=self._base_arrays(1))
        self.nwm = pd.DataFrame(data=self.negative_weight_matrix,
                                index=self._row_names(1),
                                columns=self._base_arrays(1))
        self.pwa = pd.DataFrame(data=np.concatenate([self.positive_weight_array[:, k, :] for k in range(4)],
                                                    axis=1),
                                index=self._row_names(2),
                                columns=self._base_arrays(2)
                                )
        self.nwa = pd.DataFrame(data=np.concatenate([self.negative_weight_array[:, k, :] for k in range(4)],
                                                    axis=1),
                                index=self._row_names(2),
                                columns=self._base_arrays(2)
                                )

    def _save_params(self):
        """
        保存模型权重
        """
        self.pwm.to_csv(self.model_dir + "positive_weight_matrix.csv")
        self.nwm.to_csv(self.model_dir + "negative_weight_matrix.csv")
        self.pwa.to_csv(self.model_dir + "positive_weight_array.csv")
        self.nwa.to_csv(self.model_dir + "negative_weight_array.csv")
        np.savez(self.model_dir + "weights.npz",
                 positive_weight_matrix=self.positive_weight_matrix,
                 negative_weight_matrix=self.negative_weight_matrix,
                 positive_weight_array=self.positive_weight_array,
                 negative_weight_array=self.negative_weight_array
                 )

    def load_params(self):
        """
        从文件中加载模型权重
        """
        weights_file = np.load(self.model_dir + "weights.npz")

        self.positive_weight_matrix = weights_file["positive_weight_matrix"]
        self.negative_weight_matrix = weights_file["negative_weight_matrix"]
        print("Loaded wmm weights")
        self.positive_weight_array = weights_file["positive_weight_array"]
        self.negative_weight_array = weights_file["negative_weight_array"]
        print("Loaded wam weights")

    def _plot_weights(self):
        """
        绘制模型权重热图
        """
        for weight_df, fig_name in zip([self.pwm, self.nwm, self.pwa, self.nwa],
                                       ["positive_weight_matrix.png",
                                        "negative_weight_matrix.png",
                                        "positive_weight_array.png",
                                        "negative_weight_array.png"
                                        ]):
            sns.heatmap(weight_df, cmap="viridis")
            plt.savefig(self.model_dir + fig_name, dpi=400, bbox_inches='tight')
            plt.show()

    def predict_scores(self, signals, model_type="wam"):
        """
        使用模型进行打分预测
        """
        _scores = []
        for signal in signals:
            score = 0
            if model_type == "wmm":
                for i in range(len(signal)):
                    bx = self.bases[signal[i]]
                    score += np.log(self.positive_weight_matrix[i, bx]) \
                             - np.log(self.negative_weight_matrix[i, bx])
            else:
                bx0 = self.bases[signal[0]]
                score += np.log(self.positive_weight_matrix[0, bx0]) \
                         - np.log(self.negative_weight_matrix[0, bx0])
                for i in range(1, len(signal)):
                    bx = self.bases[signal[i - 1]]
                    by = self.bases[signal[i]]
                    score += np.log(self.positive_weight_array[i - 1, bx, by]) \
                             - np.log(self.negative_weight_array[i - 1, bx, by])
            _scores.append(score)

        return np.array(_scores)

    def predict(self, signals, threshold, model_type="wam"):
        """
        使用模型进行分类预测
        """
        _scores = self.predict_scores(signals, model_type)
        _labels = np.zeros(len(signals), dtype=int)
        isGreater = _scores >= threshold
        _labels[isGreater] = 1

        return _labels


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


if __name__ == "__main__":

    start = datetime.datetime.now()

    """*********************************************准备*********************************************"""
    signals_folder = "u3d6_u12d5"
    dulen, ddlen, aulen, adlen = [int(num) for num in re.findall("\d+", signals_folder)]
    
    feat_dir = f"../Data_files/feature_data/{signals_folder}/"
    fig_dir = f"../Figures/WAM/{signals_folder}/"
    model_dir = f"../Models/WAM/{signals_folder}/"
    for path in [fig_dir, model_dir]:
        if not os.path.exists(path):
            os.makedirs(path)

    # 加载数据集
    donor_signals_training, bd_signals_training = load_signals(feat_dir, "donor", "training")
    donor_signals_testing, bd_signals_testing = load_signals(feat_dir, "donor", "testing")

    # 测试集信号
    signals_testing = donor_signals_testing + bd_signals_testing
    # 测试集标签
    donor_labels_testing = [1 for _ in range(len(donor_signals_testing))]
    bd_labels_testing = [0 for _ in range(len(bd_signals_testing))]
    labels_testing = donor_labels_testing + bd_labels_testing
    """*********************************************模型训练测试*********************************************"""
    # donor位点识别器
    donor_clf = WAM(dulen, ddlen, model_dir)
    # # 训练模型
    # donor_clf.fit(donor_signals_training, bd_signals_training)

    # 加载模型
    donor_clf.load_params()

    donor_scores_wmm = donor_clf.predict_scores(donor_signals_training, "wmm")
    donor_scores_wam = donor_clf.predict_scores(donor_signals_training, "wam")

    ratio = [i / len(donor_scores_wmm) for i in range(len(donor_scores_wmm))]
    plt.figure(figsize=(9, 6))
    plt.plot(ratio, sorted(donor_scores_wmm), 'b-', linewidth=1)
    plt.plot(ratio, sorted(donor_scores_wam), 'r', linewidth=1)
    plt.grid()
    plt.legend(["WMM score", "WAM score"], loc='best')
    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("site ratio")
    plt.ylabel("donor score")
    plt.title("Donor score distribution")
    plt.savefig(fig_dir + "donor scores.png", dpi=400, bbox_inches='tight')
    plt.show()

    """*********************************************模型测试*********************************************"""
    scores_wmm = donor_clf.predict_scores(signals_testing, "wmm")
    scores_wam = donor_clf.predict_scores(signals_testing, "wam")

    # 绘制TPR-FPR的ROC图
    fpr_wmm, tpr_wmm, thr_wmm = metrics.roc_curve(labels_testing, scores_wmm)
    fpr_wam, tpr_wam, thr_wam = metrics.roc_curve(labels_testing, scores_wam)
    auc_wmm = metrics.auc(fpr_wmm, tpr_wmm)
    auc_wam = metrics.auc(fpr_wam, tpr_wam)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plot_roc(fpr_wmm, tpr_wmm, auc_wmm, "WMM", "WMM")
    plt.subplot(1, 2, 2)
    plot_roc(fpr_wam, tpr_wam, auc_wam, "WAM", "WAM")
    plt.savefig(fig_dir + "tpr-fpr.png", dpi=400, bbox_inches="tight")
    plt.show()

    # 绘制Precision-Recall的PR图
    precision_wmm, recall_wmm, thresholds_wmm = metrics.precision_recall_curve(labels_testing, scores_wmm)
    precision_wam, recall_wam, thresholds_wam = metrics.precision_recall_curve(labels_testing, scores_wam)
    auc_wmm = metrics.auc(recall_wmm, precision_wmm)
    auc_wam = metrics.auc(recall_wam, precision_wam)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plot_pr(recall_wmm, precision_wmm, auc_wmm, "WMM", "WMM")
    plt.subplot(1, 2, 2)
    plot_pr(recall_wam, precision_wam, auc_wam, "WAM", "WAM")
    plt.savefig(fig_dir + "precision-recall.png", dpi=400, bbox_inches='tight')
    plt.show()

    # 计算f1-score最大的阈值
    max_f1score = 0
    ulti_thr = 0.5
    thrs = np.arange(min(thresholds_wam), max(thresholds_wam), 0.05)
    for thr in thrs:
        labels_pred = np.zeros(len(signals_testing), dtype=int)
        labels_pred[scores_wam >= thr] = 1
        f1score = metrics.f1_score(labels_testing, labels_pred)
        if f1score > max_f1score:
            max_f1score = f1score
            ulti_thr = thr
    print(max_f1score, ulti_thr)

    end = datetime.datetime.now()
    print(f"程序运行时间： {end - start}")
