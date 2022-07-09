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
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from sklearn import metrics
from signals_processing import load_signals
from utils import *


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
        adj_pos = [f"({position[i - 1]},{position[i]})" for i in range(1, len(position))]

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
            fig, ax = plt.subplots()
            sns.heatmap(weight_df, cmap="viridis", ax=ax)
            fig.savefig(self.model_dir + fig_name, dpi=400, bbox_inches='tight')
            plt.close()

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


if __name__ == "__main__":

    start = datetime.datetime.now()
    # site_type = "donor"
    site_type = "acceptor"

    """*****************************************基于多种信号长度的wam模型的对比******************************************"""
    # styles = ['b-', 'r:', 'm--', 'g-.', 'y-', 'c-']
    # # 设置画布
    # fig1, ax1 = plt.subplots()
    # ax1.plot([0, 1], [0, 1], 'k--', linewidth=1)
    # ax1.set_xlabel("FPR")
    # ax1.set_ylabel("TPR")
    # ax1.set_title(f"WAM ROC Curve")
    # axins1 = ax1.inset_axes([0.3, 0.3, 0.3, 0.3])
    # fig2, ax2 = plt.subplots()
    # ax2.plot([0, 1], [1, 0], 'k--', linewidth=1)
    # ax2.set_xlabel("Recall")
    # ax2.set_ylabel("Precision")
    # ax2.set_title(f"WAM PR Curve")
    #
    # fig_dir = f"../Figures/WAM/{site_type}/"
    #
    # signals_folders = ["u3d6_u15d2", "u3d9_u15d4", "u6d9_u20d4", "u6d12_u20d6", "u9d12_u25d6", "u9d15_u25d9"]
    # for signals_folder, style in zip(signals_folders, styles):
    #     dulen, ddlen, aulen, adlen = [int(num) for num in re.findall("\d+", signals_folder)]
    #
    #     feat_dir = f"../Data_files/feature_data/{signals_folder}/"
    #     model_dir = f"../Models/WAM/{site_type}/{signals_folder}/"
    #     for path in [fig_dir, model_dir]:
    #         if not os.path.exists(path):
    #             os.makedirs(path)
    #
    #     # 加载数据集
    #     pos_signals_training, neg_signals_training = load_signals(feat_dir, site_type, "training")
    #     pos_signals_testing, neg_signals_testing = load_signals(feat_dir, site_type, "testing")
    #     # 设置测试集
    #     signals_testing = pos_signals_testing + neg_signals_testing
    #     labels_testing = make_labels(pos_signals_testing, neg_signals_testing)
    #
    #     # 剪接位点识别器
    #     if site_type == "donor":
    #         splice_clf = WAM(dulen, ddlen, model_dir)
    #     else:
    #         splice_clf = WAM(aulen, adlen, model_dir)
    #     # 训练模型
    #     splice_clf.fit(pos_signals_training, neg_signals_training)
    #     # 模型预测及评估
    #     scores_testing = splice_clf.predict_scores(signals_testing)
    #     fpr, tpr, thr = metrics.roc_curve(labels_testing, scores_testing)
    #     precision, recall, thresholds = metrics.precision_recall_curve(labels_testing, scores_testing)
    #
    #     if site_type == "donor":
    #         signal_size = f"Signal=[{-dulen}, {ddlen}]"
    #     else:
    #         signal_size = f"Signal=[{-aulen}, {adlen}]"
    #
    #     # 绘制ROC曲线
    #     ax1.plot(fpr, tpr, style,
    #              label=f"{signal_size} (AUC={metrics.auc(fpr, tpr):.3f})",
    #              linewidth=0.8)
    #     axins1.plot(fpr, tpr, style)
    #     # 绘制PR曲线
    #     ax2.plot(recall, precision, style,
    #              label=f"{signal_size} (AUC={metrics.auc(recall, precision):.3f})",
    #              linewidth=0.8)
    #
    # axins1.set_xlim(0, 0.15)
    # axins1.set_ylim(0.85, 1.0)
    # mark_inset(ax1, axins1, loc1=3, loc2=1)
    # ax1.legend(loc="best", fontsize="x-small")
    # ax2.legend(loc="best", fontsize="x-small")
    # fig1.savefig(fig_dir + f"tpr-fpr.png", dpi=400, bbox_inches='tight')
    # fig2.savefig(fig_dir + f"precision-recall.png", dpi=400, bbox_inches='tight')
    # # plt.show()

    """*********************************************wmm和wam模型的对比测试*********************************************"""
    signals_folder = "u3d6_u15d2"
    # signals_folder = "u3d9_u15d4"
    # signals_folder = "u6d9_u20d4"
    # signals_folder = "u6d12_u20d6"
    # signals_folder = "u9d12_u25d6"
    # signals_folder = "u9d15_u25d9"
    dulen, ddlen, aulen, adlen = [int(num) for num in re.findall("\d+", signals_folder)]

    feat_dir = f"../Data_files/feature_data/{signals_folder}/"
    fig_dir = f"../Figures/WAM/{site_type}/{signals_folder}/"
    model_dir = f"../Models/WAM/{site_type}/{signals_folder}/"
    for path in [fig_dir, model_dir]:
        if not os.path.exists(path):
            os.makedirs(path)

    # 加载数据集
    pos_signals_training, neg_signals_training = load_signals(feat_dir, site_type, "training")
    pos_signals_testing, neg_signals_testing = load_signals(feat_dir, site_type, "testing")
    # 设置测试集
    signals_testing = pos_signals_testing + neg_signals_testing
    labels_testing = make_labels(pos_signals_testing, neg_signals_testing)

    # 剪接位点识别器
    if site_type == "donor":
        splice_clf = WAM(dulen, ddlen, model_dir)
    else:
        splice_clf = WAM(aulen, adlen, model_dir)
    # # 训练模型
    splice_clf.fit(pos_signals_training, neg_signals_training)
    # 加载模型
    # splice_clf.load_params()

    wmm_scores_training = splice_clf.predict_scores(pos_signals_training, "wmm")
    wam_scores_training = splice_clf.predict_scores(pos_signals_training, "wam")
    # 绘制训练打分分布图
    ratio = [i / len(wmm_scores_training) for i in range(len(wmm_scores_training))]
    plt.figure(figsize=(9, 6))
    plt.plot(ratio, sorted(wmm_scores_training), 'b-', linewidth=1)
    plt.plot(ratio, sorted(wam_scores_training), 'r', linewidth=1)
    plt.grid()
    plt.legend(["WMM score", "WAM score"], loc='best')
    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("site ratio")
    plt.ylabel("score")
    plt.title("Training scores distribution")
    plt.savefig(fig_dir + "training scores scores.png", dpi=400, bbox_inches='tight')
    plt.show()

    wmm_scores_testing = splice_clf.predict_scores(signals_testing, "wmm")
    wam_scores_testing = splice_clf.predict_scores(signals_testing, "wam")

    # 绘制TPR-FPR的ROC图
    wmm_fpr, wmm_tpr, wmm_thres = metrics.roc_curve(labels_testing, wmm_scores_testing)
    wam_fpr, wam_tpr, wam_thres = metrics.roc_curve(labels_testing, wam_scores_testing)
    wmm_auc = metrics.auc(wmm_fpr, wmm_tpr)
    wam_auc = metrics.auc(wam_fpr, wam_tpr)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plot_roc(wmm_fpr, wmm_tpr, wmm_auc, "WMM", "WMM")
    plt.subplot(1, 2, 2)
    plot_roc(wam_fpr, wam_tpr, wam_auc, "WAM", "WAM")
    plt.savefig(fig_dir + "tpr-fpr.png", dpi=400, bbox_inches="tight")
    plt.show()

    # 绘制Precision-Recall的PR图
    wmm_precision, wmm_recall_, wmm_thrs = metrics.precision_recall_curve(labels_testing, wmm_scores_testing)
    wam_precision, wam_recall, wam_thrs = metrics.precision_recall_curve(labels_testing, wam_scores_testing)
    wmm_auc = metrics.auc(wmm_recall_, wmm_precision)
    wam_auc = metrics.auc(wam_recall, wam_precision)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plot_pr(wmm_recall_, wmm_precision, wmm_auc, "WMM", "WMM")
    plt.subplot(1, 2, 2)
    plot_pr(wam_recall, wam_precision, wam_auc, "WAM", "WAM")
    plt.savefig(fig_dir + "precision-recall.png", dpi=400, bbox_inches='tight')
    plt.show()

    # 对于wam模型，计算最大f1-score对应的阈值
    scores_testing = splice_clf.predict_scores(signals_testing)
    find_thr(labels_testing, scores_testing)

    end = datetime.datetime.now()
    print(f"程序运行时间： {end - start}")
