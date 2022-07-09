# -*- coding: utf-8 -*-
# @Time    : 2022/5/13 15:20
# @Author  : Terence Tan
# @Email   : 2228254095@qq.com
# @FileName: gene_svm.py
# @Software: PyCharm

import os
import re
import sys
import datetime
import joblib
import umap
import numpy as np
from sklearn import svm
from sklearn import decomposition
from sklearn import manifold
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, \
    inset_axes, zoomed_inset_axes
from signals_processing import load_signals


class SVM:

    def __init__(self, signal_ulen, signal_dlen, model_par_dir, encoder_type="sparse"):
        self.bases = {'a': 0, 'c': 1, 'g': 2, 't': 3}
        self.signal_ulen = signal_ulen
        self.signal_dlen = signal_dlen
        self.signal_len = signal_ulen + signal_dlen
        self.model_par_dir = model_par_dir
        # 编码器
        self.encoder_type = encoder_type
        self.freq_mat = None
        self.signal_encoder = None
        # 分类器
        self.C = 10.0
        self.kernel = "rbf"
        self.classifier = svm.SVC(C=self.C, kernel=self.kernel)

    def fit(self, pos_signals, neg_signals):
        """
        训练模型
        """
        data, labels = self.known_encoder(pos_signals, neg_signals)
        print(data.shape)
        print(f"Fitting SVM model with {self.encoder_type} data...")
        self.classifier.fit(data, labels)

        # save model parameters
        self._save_params()

    def _save_params(self):
        """
        保存模型参数
        """
        _model_dir = self.model_par_dir + f"{self.encoder_type}/"
        if not os.path.exists(_model_dir):
            os.makedirs(_model_dir)
        joblib.dump(self.freq_mat, _model_dir + "freq_mat.pkl")
        joblib.dump(self.classifier, _model_dir + "svm_model.pkl")

    def load_params(self):
        """
        加载模型参数
        """
        _model_dir = self.model_par_dir + f"{self.encoder_type}/"
        self.freq_mat = joblib.load(_model_dir + "freq_mat.pkl")
        self.classifier = joblib.load(_model_dir + "svm_model.pkl")

    def tune_hyperparams(self, pos_signals, neg_signals):
        """
        网格搜索最优超参数
        """
        tune_parameters = [
            # {'kernel': ['linear'], 'C': [1, 10, 100]},
            {'kernel': ['poly'], 'degree': [3, 4, 5], 'C': [1, 10, 100]},
            {'kernel': ['rbf'], 'C': [1, 10, 100]},
            # {'kernel': ['sigmoid'], 'C': [1, 10, 100]}
        ]
        # 使用f1-score作为模型性能评估指标，综合考虑了precision和recall
        clf = GridSearchCV(svm.SVC(), tune_parameters, scoring="f1", n_jobs=7, cv=5, verbose=3)

        data, labels = self.known_encoder(pos_signals, neg_signals)
        clf.fit(data, labels)
        print("Best parameters: ", end="")
        print(clf.best_params_)
        print("Grid scores: ")

        # 具体的参数间不同数值组合后得到的分数
        for mean, std, rank, params in zip(
                clf.cv_results_["mean_test_score"],
                clf.cv_results_["std_test_score"],
                clf.cv_results_["rank_test_score"],
                clf.cv_results_['params'],
        ):
            print(f"\trank: {rank} \t test_f1_score: {mean:.3f}(+/-{std * 3:.3f}) for {params}")

    def _compute_freq(self, pos_signals, neg_signals):
        if self.encoder_type == "sparse":
            self.freq_mat = None
        elif self.encoder_type == "mm1":
            self.freq_mat = self._compute_cpm(pos_signals, 1)
        elif self.encoder_type == "mm2":
            self.freq_mat = self._compute_cpm(pos_signals, 2)
        else:
            self.freq_mat = self._compute_fdtf(pos_signals, neg_signals)

    def single_encoder(self):
        """
        信号编码
        """
        if self.encoder_type == "sparse":
            encoder = lambda signal: self.sparse_encoder(signal, self.freq_mat)
        elif self.encoder_type == "mm1":
            encoder = lambda signal: self.mm1_encoder(signal, self.freq_mat)
        elif self.encoder_type == "mm2":
            encoder = lambda signal: self.mm2_encoder(signal, self.freq_mat)
        else:
            encoder = lambda signal: self.fdtf_encoder(signal, self.freq_mat)

        self.signal_encoder = lambda signals: np.array([encoder(signal) for signal in signals])

    def known_encoder(self, pos_signals, neg_signals):
        """
        已知类型数据编码
        """
        self._compute_freq(pos_signals, neg_signals)
        self.single_encoder()
        # 数据
        pos_data = self.signal_encoder(pos_signals)
        neg_data = self.signal_encoder(neg_signals)
        data = np.vstack((pos_data, neg_data))
        # 标签
        pos_labels = np.array([1 for _ in range(pos_data.shape[0])])
        neg_labels = np.array([-1 for _ in range(neg_data.shape[0])])
        labels = np.hstack((pos_labels, neg_labels))

        return data, labels

    def unknown_encoder(self, signals):
        """
        未知类型数据编码
        """
        self.single_encoder()
        data = self.signal_encoder(signals)

        return data

    def sparse_encoder(self, signal, cpm=None):
        """
        Sparse encoding
        return a vector of length 4 * L
        """
        vec = [0 for _ in signal] * 4
        for idx, aa in enumerate(signal):
            vec[idx * 4 + self.bases[aa]] = 1
        return vec

    def _compute_cpm(self, pos_signals, order=1):
        """
        order = 1:
        Compute the conditional probability matrix of order 16 × (K-1)
        using the positive splice signals in training set
        order = 2:
        Compute the conditional probability matrix of order 64 × (K-2)
        using the positive splice signals training set
        """
        # 16 × (K - 1)
        if order == 1:
            ppm = np.zeros((self.signal_len - 1, 4), dtype=np.float64)  # mononucleotide频率
            jpm = np.zeros((self.signal_len - 1, 4, 4), dtype=np.float64)  # dinucleotide频率
            for signal in pos_signals:
                for i in range(1, len(signal)):
                    bx = self.bases[signal[i - 1].lower()]
                    by = self.bases[signal[i].lower()]
                    ppm[i - 1, bx] += 1.0
                    jpm[i - 1, bx, by] += 1.0
            ppm = np.tile(np.expand_dims((ppm / len(pos_signals)), -1), 4)
            ppm[ppm == 0] = 1e-6  # 填充概率为0的位置
            jpm = jpm / len(pos_signals)
            cpm = jpm / ppm  # 条件概率
        # 64 × (K-2)
        else:
            ppm = np.zeros((self.signal_len - 2, 4, 4), dtype=np.float64)  # dinucleotide频率
            jpm = np.zeros((self.signal_len - 2, 4, 4, 4), dtype=np.float64)  # trinucleotide频率
            for signal in pos_signals:
                for i in range(2, len(signal)):
                    bx = self.bases[signal[i - 2].lower()]
                    by = self.bases[signal[i - 1].lower()]
                    bz = self.bases[signal[i].lower()]
                    ppm[i - 2, bx, by] += 1.0
                    jpm[i - 2, bx, by, bz] += 1.0
            ppm = np.tile(np.expand_dims((ppm / len(pos_signals)), -1), 4)
            ppm[ppm == 0] = 1e-6  # 填充概率为0的位置
            jpm = jpm / len(pos_signals)
            cpm = jpm / ppm  # 条件概率

        return cpm

    def mm1_encoder(self, signal, cpm):
        """
        First order Markov model encoding
        """
        vec = np.zeros((len(signal) - 1, 4, 4))
        for i in range(1, len(signal)):
            bx = self.bases[signal[i - 1].lower()]
            by = self.bases[signal[i].lower()]
            vec[i - 1, bx, by] = cpm[i - 1, bx, by]
        vec = vec.reshape(-1)

        return vec

    def mm2_encoder(self, signal, cpm):
        """
        Second order Markov model encoding
        """
        vec = np.zeros((len(signal) - 1, 4, 4, 4))
        for i in range(2, len(signal)):
            bx = self.bases[signal[i - 2].lower()]
            by = self.bases[signal[i - 1].lower()]
            bz = self.bases[signal[i].lower()]
            vec[i - 2, bx, by, bz] = cpm[i - 2, bx, by, bz]
        vec = vec.reshape(-1)

        return vec

    def _compute_fdtf(self, pos_signals, neg_signals):
        """
        A FDTF-coding matrix with dimension 16*(k-1) is obtained by subtract-
        ing the true donor-coding matrices from the false donor-coding matrices.
        """
        tppm = np.zeros((self.signal_len - 1, 4, 4), dtype=np.float64)
        fppm = np.zeros((self.signal_len - 1, 4, 4), dtype=np.float64)
        for signal in pos_signals:
            for i in range(1, len(signal)):
                bx = self.bases[signal[i - 1].lower()]
                by = self.bases[signal[i].lower()]
                tppm[i - 1, bx, by] += 1.0
        tppm = tppm / len(pos_signals)
        for signal in neg_signals:
            for i in range(1, len(signal)):
                bx = self.bases[signal[i - 1].lower()]
                by = self.bases[signal[i].lower()]
                fppm[i - 1, bx, by] += 1.0
        fppm = fppm / len(neg_signals)

        return tppm - fppm

    def fdtf_encoder(self, signal, ftdf):
        """
        FDTF(frequency difference between the true sites and false site) encoding
        """
        vec = np.zeros((len(signal) - 1, 4, 4))
        for i in range(1, len(signal)):
            bx = self.bases[signal[i - 1].lower()]
            by = self.bases[signal[i].lower()]
            vec[i - 1, bx, by] = ftdf[i - 1, bx, by]
        vec = vec.reshape(-1)

        return vec

    def predict_scores(self, signals):
        """
        SVM模型打分预测
        """
        data = self.unknown_encoder(signals)
        _scores = self.classifier.decision_function(data)

        return _scores

    def predict(self, signals):
        """
        SVM模型分类预测
        """
        data = self.unknown_encoder(signals)
        _labels = self.classifier.predict(data)

        return _labels

    def accuracy_score(self, signals, labels):
        """
        SVM模型分类预测准确度打分
        """
        data = self.unknown_encoder(signals)
        _acc = self.classifier.score(data, labels)

        return _acc


if __name__ == "__main__":
    start = datetime.datetime.now()

    """*********************************************数据准备*********************************************"""
    signals_folder = "u3d6_u12d5"
    # signals_folder = "u6d9_u15d6"
    # signals_folder = "u9d12_u20d9"
    # signals_folder = "u9d15_u27d9"

    dulen, ddlen, aulen, adlen = [int(num) for num in re.findall("\d+", signals_folder)]

    feat_dir = f"../Data_files/feature_data/{signals_folder}/"
    fig_par_dir = f"../Figures/SVM/{signals_folder}/"
    model_par_dir = f"../Models/SVM/{signals_folder}/"

    enc_types = ["sparse", "mm1", "mm2", "fdtf"]
    for enc_type in enc_types:
        fig_dir = fig_par_dir + f"{enc_type}/"
        model_dir = model_par_dir + f"{enc_type}/"
        for path in [fig_dir, model_dir]:
            if not os.path.exists(path):
                os.makedirs(path)

    # 加载数据集
    donor_signals_training, bd_signals_training = load_signals(feat_dir, "donor", "training")
    donor_signals_testing, bd_signals_testing = load_signals(feat_dir, "donor", "testing")

    # # 抽取部分数据用于代码测试
    # p = 2000
    # n = 20000
    # donor_signals_training = donor_signals_training[:p]
    # bd_signals_training = bd_signals_training[:n]
    # donor_signals_testing = donor_signals_testing[:p]
    # bd_signals_testing = bd_signals_testing[:n]
    
    # 测试集信号
    signals_testing = donor_signals_testing + bd_signals_testing
    # 测试集标签
    donor_labels_testing = np.array([1 for _ in range(len(donor_signals_testing))])
    bd_labels_testing = np.array([-1 for _ in range(len(bd_signals_testing))])
    labels_testing = np.hstack((donor_labels_testing, bd_labels_testing))
    
    """************************************多种编码方式对比************************************"""
    # donor_sparse_svm = SVM(dulen, ddlen, model_par_dir, "sparse")
    # donor_mm1_svm = SVM(dulen, ddlen, model_par_dir, "mm1")
    # donor_mm2_svm = SVM(dulen, ddlen, model_par_dir, "mm2")
    # donor_fdtf_svm = SVM(dulen, ddlen, model_par_dir, "fdtf")
    #
    # donor_svms = [donor_sparse_svm, donor_mm1_svm, donor_mm2_svm, donor_fdtf_svm]
    #
    # for svm_model in donor_svms:
    #     # # 网格搜索最优超参数
    #     # svm_model.tune_hyperparams(donor_signals_training, bd_signals_training)
    #
    #     # # 模型训练
    #     svm_model.fit(donor_signals_training, bd_signals_training)
    #
    #     # 模型加载
    #     # svm_model.load_params()
    #
    # styles = ['b-', 'r:', 'm--', 'g-.']
    # # 设置画布
    # fig1, ax1 = plt.subplots()
    # ax1.plot([0, 1], [0, 1], 'k--', linewidth=1)
    # ax1.set_xlabel("FPR")
    # ax1.set_ylabel("TPR")
    # ax1.set_title(f"SVM ROC Curve")
    # axins1 = ax1.inset_axes([0.3, 0.3, 0.3, 0.3])
    # fig2, ax2 = plt.subplots()
    # ax2.plot([0, 1], [1, 0], 'k--', linewidth=1)
    # ax2.set_xlabel("Recall")
    # ax2.set_ylabel("Precision")
    # ax2.set_title(f"SVM PR Curve")
    #
    # # 测试集预测
    # for svm_model, style in zip(donor_svms, styles):
    #     labels_pred = svm_model.predict(signals_testing)
    #     scores_pred = svm_model.predict_scores(signals_testing)
    #
    #     fpr, tpr, thr = metrics.roc_curve(labels_testing, scores_pred)
    #     precision, recall, thresholds = metrics.precision_recall_curve(labels_testing, scores_pred)
    #
    #     # 绘制ROC曲线
    #     ax1.plot(fpr, tpr, style,
    #              label=f"{svm_model.encoder_type} (AUC={metrics.auc(fpr, tpr):.3f})", linewidth=1)
    #     axins1.plot(fpr, tpr, style)
    #     # 绘制PR曲线
    #     ax2.plot(recall, precision, style,
    #              label=f"{svm_model.encoder_type} (AUC={metrics.auc(recall, precision):.3f})", linewidth=1)
    #
    #     # print(f"precision={metrics.precision_score(labels_testing, labels_pred):.3f}")
    #     # print(f"recall={metrics.recall_score(labels_testing, labels_pred):.3f}")
    #     # print(f"accuracy={svm_model.accuracy_score(signals_testing, labels_testing):.3f}")
    #     # print(f"f1-score={metrics.f1_score(labels_testing, labels_pred):.3f}")
    #     # print(classification_report(labels_testing, labels_pred))  # precision/recall/f1-score
    #
    # axins1.set_xlim(0, 0.1)
    # axins1.set_ylim(0.85, 0.95)
    # mark_inset(ax1, axins1, loc1=3, loc2=1)
    # ax1.legend(loc="lower right")
    # ax2.legend(loc="lower left")
    # fig1.savefig(fig_par_dir + f"tpr-fpr.png", dpi=400, bbox_inches='tight')
    # fig2.savefig(fig_par_dir + f"precision-recall.png", dpi=400, bbox_inches='tight')
    # plt.show()

    """************************************基于稀疏编码的模型************************************"""
    donor_svm = SVM(dulen, ddlen, model_par_dir, "sparse")
    donor_svm.fit(donor_signals_training, bd_signals_training)
    print("模型训练完成")
    # # 模型加载
    # donor_svm.load_params()
    # print("模型加载完成")

    fig1, ax1 = plt.subplots()
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax1.set_xlabel("FPR")
    ax1.set_ylabel("TPR")
    ax1.set_title(f"SVM ROC Curve")
    fig2, ax2 = plt.subplots()
    ax2.plot([0, 1], [1, 0], 'k--', linewidth=1)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title(f"SVM PR Curve")

    scores_pred = donor_svm.predict_scores(signals_testing)
    print("模型打分完成")
    fpr, tpr, thr = metrics.roc_curve(labels_testing, scores_pred)
    precision, recall, thresholds = metrics.precision_recall_curve(labels_testing, scores_pred)

    # 绘制ROC曲线
    ax1.plot(fpr, tpr,
             label=f"{donor_svm.encoder_type} (AUC={metrics.auc(fpr, tpr):.3f})", linewidth=1)
    # 绘制PR曲线
    ax2.plot(recall, precision,
             label=f"{donor_svm.encoder_type} (AUC={metrics.auc(recall, precision):.3f})", linewidth=1)

    ax1.legend(loc="lower right")
    ax2.legend(loc="lower left")
    fig1.savefig(fig_par_dir + f"{donor_svm.encoder_type}/tpr-fpr.png", dpi=400, bbox_inches='tight')
    fig2.savefig(fig_par_dir + f"{donor_svm.encoder_type}/precision-recall.png", dpi=400, bbox_inches='tight')
    plt.show()

    """************************************降维分析************************************"""
    # data_training, labels_training = donor_sparse_svm.known_encoder(donor_signals_training, bd_signals_training)
    # # PCA降维分析
    # pca = decomposition.PCA(n_components=2)
    # data_r1_training = pca.fit_transform(data_training)
    # print(
    #     "explained variance ratio (first two components): %s"
    #     % str(pca.explained_variance_ratio_)
    # )
    # plt.figure()
    # colors = ["cyan", "red"]
    # classes = ["negative", "positive"]
    # for color, i, cls in zip(colors, [-1, 1], classes):
    #     plt.scatter(data_r1_training[labels_training == i, 0],
    #                 data_r1_training[labels_training == i, 1],
    #                 s=2,
    #                 c=color,
    #                 alpha=0.6,
    #                 label=cls
    #                 )
    # plt.legend(loc="best", shadow=False, scatterpoints=80)
    # plt.title("PCA of donor training set")
    # plt.savefig(fig_par_dir+"pca_donor_training.png", dpi=400, bbox_inches='tight')
    # plt.show()
    #
    # # LDA降维分析
    # lda = LinearDiscriminantAnalysis(n_components=1)
    # data_r2_training = lda.fit_transform(data_training, labels_training)
    # plt.figure()
    # colors = ["cyan", "red"]
    # classes = ["negative", "positive"]
    # for color, i, cls in zip(colors, [-1, 1], classes):
    #     plt.scatter(data_r2_training[labels_training == i],
    #                 labels_training[labels_training == i],
    #                 s=2,
    #                 c=color,
    #                 alpha=0.6,
    #                 label=cls
    #                 )
    # plt.legend(loc="best", shadow=False, scatterpoints=80)
    # plt.title("LDA of donor training set")
    # plt.savefig(fig_par_dir+"lda_donor_training.png", dpi=400, bbox_inches='tight')
    # plt.show()
    #
    # # NMF降维分析
    # nmf = decomposition.NMF(n_components=2, init='random', max_iter=1000)
    # data_r3_training = nmf.fit_transform(data_training)
    # plt.figure()
    # colors = ["cyan", "red"]
    # classes = ["negative", "positive"]
    # for color, i, cls in zip(colors, [-1, 1], classes):
    #     plt.scatter(data_r3_training[labels_training == i, 0],
    #                 data_r3_training[labels_training == i, 1],
    #                 s=2,
    #                 c=color,
    #                 alpha=0.6,
    #                 label=cls
    #                 )
    # plt.legend(loc="best", shadow=False, scatterpoints=80)
    # plt.title("NMF of donor training set")
    # plt.savefig(fig_par_dir+"nmf_donor_training.png", dpi=400, bbox_inches='tight')
    # plt.show()
    #
    # # t-SNE降维分析
    # tsne = manifold.TSNE(n_components=2,
    #                      learning_rate='auto',
    #                      init='random',
    #                      n_iter=1000,
    #                      n_jobs=7)
    # data_r4_training = tsne.fit_transform(data_training)
    # plt.figure()
    # colors = ["cyan", "red"]
    # classes = ["negative", "positive"]
    # for color, i, cls in zip(colors, [-1, 1], classes):
    #     plt.scatter(data_r4_training[labels_training == i, 0],
    #                 data_r4_training[labels_training == i, 1],
    #                 s=2,
    #                 c=color,
    #                 alpha=0.6,
    #                 label=cls
    #                 )
    # plt.legend(loc="best", shadow=False, scatterpoints=80)
    # plt.title("t-SNE of donor training set")
    # plt.savefig(fig_par_dir+"tsne_donor_training.png", dpi=400, bbox_inches='tight')
    # plt.show()
    #
    # # umap降维分析
    # embedding = umap.UMAP(n_neighbors=15,
    #                       n_components=2,
    #                       metric="euclidean",
    #                       random_state=2022,
    #                       )
    # data_r5_training = embedding.fit_transform(data_training)
    # plt.figure()
    # colors = ["cyan", "red"]
    # classes = ["negative", "positive"]
    # for color, i, cls in zip(colors, [-1, 1], classes):
    #     plt.scatter(data_r5_training[labels_training == i, 0],
    #                 data_r5_training[labels_training == i, 1],
    #                 s=2,
    #                 c=color,
    #                 alpha=0.6,
    #                 label=cls
    #                 )
    # plt.legend(loc="best", shadow=False, scatterpoints=80)
    # plt.title("UMAP of donor training set")
    # plt.savefig(fig_par_dir+"umap_donor_training.png", dpi=400, bbox_inches='tight')
    # plt.show()

    end = datetime.datetime.now()
    print(f"程序运行时间： {end - start}")
