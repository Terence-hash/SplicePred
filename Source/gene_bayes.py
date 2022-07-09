# -*- coding: utf-8 -*-
# @Time    : 2022/4/19 15:57
# @Author  : Terence Tan
# @Email   : 2228254095@qq.com
# @FileName: gene_bayes.py
# @Software: PyCharm

import os
import re
import sys
import datetime
import joblib
from pprint import pprint
import math
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import chi2_contingency
from sklearn import metrics
from sklearn.metrics import classification_report
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.estimators import BicScore, K2Score, BDeuScore, BDsScore
from pgmpy.estimators import PC, HillClimbSearch, MmhcEstimator
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, \
    inset_axes, zoomed_inset_axes
import matplotlib.pyplot as plt
from signals_processing import load_signals


class BN:

    def __init__(self, signal_ulen, signal_dlen, model_par_dir, net_struc="hillclimb"):
        self.bases = {'a': 0, 'c': 1, 'g': 2, 't': 3}
        self.signal_ulen = signal_ulen
        self.signal_dlen = signal_dlen
        self.signal_len = signal_ulen + signal_dlen
        self.model_par_dir = model_par_dir
        self.pri_pros = {"positive": 0.5, "negative": 0.5}

        self.net_struc = net_struc
        self.positive_network = BayesianNetwork()
        self.negative_network = BayesianNetwork()

    def build(self, signals):
        """
        构建贝叶斯网络
        """
        if self.net_struc == "naive":
            return self.build_naive_network()
        elif self.net_struc == "chain":
            return self.build_chain_network()
        elif self.net_struc == "expanded":
            return self.build_expanded_network(signals)
        else:
            return self.build_hillclimb_network(signals)

    def compute_pri_pros(self, pos_signals, neg_signals):
        """
        计算正类与父类的先验概率
        """
        self.pri_pros["positive"] = len(pos_signals) / (len(pos_signals + neg_signals))
        self.pri_pros["negative"] = len(neg_signals) / (len(pos_signals + neg_signals))

    def fit(self, pos_signals, neg_signals):
        """
        训练贝叶斯网络
        """
        self.compute_pri_pros(pos_signals, neg_signals)

        # keep the structure of positive network and negative network consistent
        self.positive_network = self.build(pos_signals)
        self.negative_network = self.positive_network.copy()

        if len(pos_signals) > 0:
            print(f"Fitting Bayesian {self.net_struc} network with positive data...")
            pos_data = self.signal_encoder(pos_signals, self.positive_network)
            self.positive_network.fit(pos_data, estimator=BayesianEstimator)
        if len(neg_signals) > 0:
            print(f"Fitting Bayesian {self.net_struc} network with negative data...")
            neg_data = self.signal_encoder(neg_signals, self.negative_network)
            self.negative_network.fit(neg_data, estimator=BayesianEstimator)

        # plot Bayesian network
        self._plot_networks()
        # save Bayesian network parameters
        self._save_params()

    def _save_params(self):
        """
        保存网络参数
        """
        _model_dir = self.model_par_dir + f"{self.net_struc}/"
        if not os.path.exists(_model_dir):
            os.makedirs(_model_dir)
        joblib.dump(self.pri_pros, _model_dir + "pri_pros.pkl")
        joblib.dump(self.positive_network, _model_dir + "positive_network.pkl")
        joblib.dump(self.negative_network, _model_dir + "negative_network.pkl")

    def load_params(self):
        """
        从保存文件中加载网络参数
        """
        _model_dir = self.model_par_dir + f"{self.net_struc}/"
        self.pri_pros = joblib.load(_model_dir + "pri_pros.pkl")
        self.positive_network = joblib.load(_model_dir + "positive_network.pkl")
        self.negative_network = joblib.load(_model_dir + "negative_network.pkl")
        print(f"Loaded Bayesian {self.net_struc} network parameters")

    def _plot_networks(self):
        """
        绘制网络结构图
        """
        _model_dir = self.model_par_dir + f"{self.net_struc}/"
        if not os.path.exists(_model_dir):
            os.makedirs(_model_dir)

        G = nx.DiGraph()
        G.add_nodes_from(self.positive_network.nodes)
        G.add_edges_from(self.positive_network.edges)

        if self.net_struc == "naive":
            pos = nx.shell_layout(G)
            fig_size = (4, 4)
            node_size = 800
            font_size = 15
        elif self.net_struc == "chain":
            pos = {node: [int(node), int(node) % 2] for node in G.nodes}
            fig_size = (6, 2)
            node_size = 800
            font_size = 15
        elif self.net_struc == "expanded":
            loc = lambda p: int(p[:p.find("(")])
            pos = {node:
                       [int(node[-2]), -(loc(node) + int(self.signal_ulen - 1))
                       if loc(node) > 0
                       else -(loc(node) + int(self.signal_ulen + 2))]
                   for node in G.nodes}
            fig_size = (8, 5)
            node_size = 800
            font_size = 10
        else:
            pos = nx.shell_layout(G)
            fig_size = (6, 3)
            node_size = 800
            font_size = 15

        plt.figure(figsize=fig_size)
        nx.draw(G, pos,
                with_labels=True,
                node_size=node_size,
                node_color="skyblue",
                node_shape="o",
                alpha=0.8,
                width=0.5,
                font_size=font_size
                )
        plt.savefig(_model_dir + "network.png", dpi=400, bbox_inches='tight')
        plt.show()

    def signal_encoder(self, signals, network=None):
        """
        将信号编码为贝叶斯网络的输入数据
        """
        if network is None:
            data = [list(i) for i in signals]
            columns = list(map(str, list(range(-self.signal_ulen, 0)) +
                               list(range(1, self.signal_dlen + 1))))
            return pd.DataFrame(data, columns=columns)

        if self.net_struc == "expanded":
            loc = lambda p: int(p[:p.find("(")])
        else:
            loc = lambda p: int(p)
        data = [[signal[loc(node) + (self.signal_ulen - 1)]
                 if loc(node) > 0
                 else signal[loc(node) + self.signal_ulen]
                 for node in network.nodes]
                for signal in signals]

        return pd.DataFrame(data, columns=network.nodes)

    def get_contingency_tables(self, signals):
        """
        计算不同位置碱基的 4×4 列联表
        """
        ctas = np.zeros((self.signal_len, self.signal_len, 4, 4), dtype=np.float64)
        for signal in signals:
            for i in range(self.signal_len):
                for j in range(self.signal_len):
                    bx = self.bases[signal[i].lower()]
                    by = self.bases[signal[j].lower()]
                    ctas[i, j, bx, by] += 1.0

        return ctas

    def dependency_test(self, signals, threshold):
        """
        不同位置碱基的依赖性卡方检验
        """
        contingency_tables = self.get_contingency_tables(signals)
        chi2stats = []
        for i in range(contingency_tables.shape[0]):
            for j in range(i + 1, contingency_tables.shape[1]):
                # 通过try-except可以排除GT(AG)
                try:
                    kf = chi2_contingency(contingency_tables[i][j])
                    p1 = i - self.signal_ulen if i <= (self.signal_ulen - 1) else i - (self.signal_ulen - 1)
                    p2 = j - self.signal_ulen if j <= (self.signal_ulen - 1) else j - (self.signal_ulen - 1)
                    chi2stats.append({"pair": (p1, p2), "p": kf[1], "chi2": kf[0], "freedom": kf[2]})
                except:
                    pass

        chi2stats = sorted(chi2stats, key=lambda r: r['chi2'], reverse=True)
        chi2stats = [chi2stat for chi2stat in chi2stats if chi2stat['p'] < threshold]

        self._plot_dependency(chi2stats)

        return chi2stats

    def _plot_dependency(self, chi2stats):
        """
        绘制节点依赖图
        """
        plt.figure(figsize=(6, 3))
        G = nx.Graph()
        edges = [link["pair"] for link in chi2stats]
        edge_labels = {link["pair"]: f"[{order + 1}] p={link['p']:.2e}"
                       for order, link in enumerate(chi2stats)}
        G.add_edges_from(edges)
        pos = {node: [int(node / 2), node % 3] for node in G.nodes}
        nx.draw(G, pos,
                with_labels=True,
                node_size=800,
                node_color="skyblue",
                node_shape="o",
                alpha=0.8,
                width=0.5,
                font_size=15,
                arrows=False
                )
        nx.draw_networkx_edge_labels(G, pos,
                                     edge_labels=edge_labels,
                                     font_size=6,
                                     font_color="k",
                                     alpha=0.8,
                                     rotate=True)
        plt.savefig(self.model_par_dir + "dependency_graph.png", dpi=400, bbox_inches='tight')
        plt.show()

    def get_adj(self, signals):
        """
        统计节点的邻近节点及卡方值，并对卡方值求和
        """
        chi2stats = self.dependency_test(signals, 1e-8)
        # 二维字典node_adj，外层字典key为各个节点，内层字典key为依赖节点的卡方值、卡方值总和
        node_adj = {pos if pos < 0 else pos:
                        {"chi2": [],
                         "chi2_sum": 0
                         }
                    for pos in list(range(-self.signal_ulen, 0)) + list(range(3, self.signal_dlen + 1))
                    }
        for chi2stat in chi2stats:
            pair = chi2stat["pair"]
            for i in range(2):
                node_adj[pair[i]]["chi2"].append((pair[1 - i], chi2stat["chi2"]))
                node_adj[pair[i]]["chi2_sum"] += chi2stat["chi2"]

        return node_adj

    def graph_expand(self, nodes, tags):
        """
        扩展依赖图形成贝叶斯网络
        """
        root = max(nodes.items(), key=lambda x: x[1]["chi2_sum"])
        layer0 = root[0]  # 根节点（第0层）
        layer1 = [node[0] for node in root[1]["chi2"]]  # 第1层
        layers = [layer0, sorted(layer1)]
        # 建立根节点与第1层之间的连接
        edge0 = {layer0: [node for node in layer1]}
        edges = [edge0]
        # 更新tags
        for key in edge0:
            for node in edge0[key]:
                tags[node][key][0] += 1

        while True:
            # 第2层节点（二维列表）
            layer2 = [list(tags[node].keys()) for node in layer1]
            # 第2层节点对应的第1层父节点
            parent = {}
            for i in range(len(layer2)):
                for node in layer2[i]:
                    pnode = layer1[i]
                    if node not in parent:
                        parent[node] = [[pnode, tags[node][pnode]]]
                    else:
                        parent[node].append([pnode, tags[node][pnode]])
            # 筛选出tags最大的2个父节点
            for key in parent.keys():
                parent[key].sort(key=lambda x: (x[1][0], -x[1][1]))
                parent[key] = parent[key][:2]
            # 建立第1层与第2层之间的连接
            edge1 = {}
            for node in parent:
                for par in parent[node]:
                    if par[0] not in edge1:
                        edge1[par[0]] = [node]
                    else:
                        edge1[par[0]].append(node)
            # 更新第2层节点以第1层节点为父节点的tags值
            for key in edge1:
                for node in edge1[key]:
                    tags[node][key][0] += 1
            # 更新layers与edges
            layer1 = sorted(list(set([nod for nods in layer2 for nod in nods])))
            layers.append(layer1)
            edges.append(edge1)
            # 依赖图中的所有连接都被实现，即可退出
            label = 0
            for value1 in tags.values():
                for value2 in value1.values():
                    if value2[0] == 0:
                        label = 1
                        break
            if label == 0:
                break

        return layers, edges

    def build_naive_network(self):
        """
        构建朴素贝叶斯模型
        """
        # 网络结构中不包含G(1)T(2)节点的情况
        net_nodes = [str(i) for i in range(-self.signal_ulen, self.signal_len - (self.signal_ulen - 1))
                     if i not in [0, 1, 2]]
        # 网络结构中包含G(1)T(2)节点的情况
        # net_nodes = [str(i) for i in range(-self.signal_ulen, self.signal_len - (self.signal_ulen - 1))
        #              if i not in [0]]
        network = BayesianNetwork()
        network.add_nodes_from(net_nodes)
        return network

    def build_chain_network(self):
        """
        构建链式贝叶斯网络
        """
        # 网络结构中不包含G(1)T(2)节点的情况
        net_nodes = [str(i) for i in range(-self.signal_ulen, self.signal_len - (self.signal_ulen - 1))
                     if i not in [0, 1, 2]]
        # 网络结构中包含G(1)T(2)节点的情况
        # net_nodes = [str(i) for i in range(-self.signal_ulen, self.signal_len - (self.signal_ulen - 1))
        #              if i not in [0]]
        net = [(net_nodes[j], net_nodes[j + 1]) for j in range(len(net_nodes) - 1)]

        return BayesianNetwork(net)

    def build_expanded_network(self, signals):
        """
        构建扩展贝叶斯网络
        """
        adj_nodes = self.get_adj(signals)
        # 二维字典，外层字典key为节点，内层字典key为对应父节点，value为[父节点使用次数、卡方统计量]
        tags = {key: {chi2[0]: [0, chi2[1]] for chi2 in adj_nodes[key]["chi2"]}
                for key in adj_nodes.keys()}
        # 扩展依赖图
        layers, edges = self.graph_expand(adj_nodes, tags)
        # 网络构建
        net = []
        for depth in range(len(edges)):
            for node in edges[depth].items():
                for child in node[1]:
                    net.append((f"{node[0]}(L{depth})", f"{child}(L{depth + 1})"))

        return BayesianNetwork(net)

    def build_hillclimb_network(self, signals):
        """
        构建最佳贝叶斯网络结构（利用爬山算法进行学习）
        """
        data = self.signal_encoder(signals)
        hc = HillClimbSearch(data=data)
        best_model = hc.estimate(scoring_method=BDsScore(data))

        return BayesianNetwork(best_model.edges)

    def predict_scores(self, signals):
        """
        使用贝叶斯网络进行打分预测
        """
        _scores = []
        data = self.signal_encoder(signals, self.positive_network)
        bnts = [self.positive_network, self.negative_network]
        prior_pros = self.pri_pros

        for row_index, row_data in data.iterrows():
            post_pros = {"positive": 0.5, "negative": 0.5}
            for bnt, pri_pro, label in zip(bnts, prior_pros.keys(), post_pros.keys()):
                for node in bnt.nodes:
                    values = bnt.get_cpds(node).values
                    state_names = bnt.get_cpds(node).state_names.keys()
                    # 网络结构中不包含G(1)T(2)节点的情况
                    index = [self.bases[row_data[col]] for col in state_names]
                    # 网络结构中包含G(1)T(2)节点的情况
                    # index = [self.bases[row_data[col]] if col not in ['1', '2'] else 0 for col in state_names]
                    post_pros[label] += math.log(eval(f"values{index}"))
                # 考虑先验概率
                # post_pros[label] += math.log(pri_pro)
            _scores.append(post_pros["positive"] - post_pros["negative"])

        return np.array(_scores)

    def predict(self, signals, threshold):
        """
        使用贝叶斯网络进行分类预测
        """
        _scores = self.predict_scores(signals)
        labels = np.zeros(len(signals), dtype=np.int)
        isGreater = _scores >= threshold
        labels[isGreater] = 1

        return labels


if __name__ == "__main__":
    start = datetime.datetime.now()

    """*********************************************数据准备*********************************************"""
    # signals_folder = "u3d6_u12d5"
    # signals_folder = "u6d9_u15d6"
    signals_folder = "u9d12_u20d9"

    dulen, ddlen, aulen, adlen = [int(num) for num in re.findall("\d+", signals_folder)]

    feat_dir = f"../Data_files/feature_data/{signals_folder}/"
    fig_par_dir = f"../Figures/Bayes/{signals_folder}/"
    model_par_dir = f"../Models/Bayes/{signals_folder}/"

    # net_strucs = ["naive", "chain", "expanded", "hillclimb"]
    net_strucs = ["naive", "chain", "hillclimb"]
    for net_struc in net_strucs:
        fig_dir = fig_par_dir + f"{net_struc}/"
        model_dir = model_par_dir + f"{net_struc}/"
        for path in [fig_dir, model_dir]:
            if not os.path.exists(path):
                os.makedirs(path)

    # 加载数据集
    donor_signals_training, bd_signals_training = load_signals(feat_dir, "donor", "training")
    donor_signals_testing, bd_signals_testing = load_signals(feat_dir, "donor", "testing")
    
    # 测试集信号
    signals_testing = donor_signals_testing + bd_signals_testing
    # 测试集标签
    donor_labels_testing = np.array([1 for _ in range(len(donor_signals_testing))])
    bd_labels_testing = np.array([0 for _ in range(len(bd_signals_testing))])
    labels_testing = np.hstack((donor_labels_testing, bd_labels_testing))
    
    """************************************多种网络结构对比************************************"""
    donor_naive_network = BN(dulen, ddlen, model_par_dir, "naive")
    donor_chain_network = BN(dulen, ddlen, model_par_dir, "chain")
    # donor_expanded_network = BN(dulen, ddlen, model_par_dir, "expanded")
    donor_hillclimb_network = BN(dulen, ddlen, model_par_dir, "hillclimb")

    donor_networks = [donor_naive_network,
                      donor_chain_network,
                      # donor_expanded_network,
                      donor_hillclimb_network
                      ]

    for network in donor_networks:
        # 网络训练
        network.fit(donor_signals_training, bd_signals_training)
        # 网络加载
        # network.load_params()

    # # 测试集预测
    # for network in donor_networks:
    #     # if network.net_struc != "hillclimb":
    #     #     continue
    #     scores_testing = network.predict_scores(signals_testing)
    #     fpr, tpr, thr = metrics.roc_curve(labels_testing, scores_testing)
    #     precision, recall, thresholds = metrics.precision_recall_curve(labels_testing, scores_testing)
    #
    #     # 绘制ROC曲线
    #     plt.figure()
    #     plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    #     plt.plot(fpr, tpr, 'b-', label=f"Bayes(AUC={metrics.auc(fpr, tpr):.3f})", linewidth=1)
    #     plt.legend(loc="best")
    #     plt.xlabel("FPR")
    #     plt.ylabel("TPR")
    #     plt.title(f"Bayes ROC Curve")
    #     plt.savefig(fig_par_dir + f"{network.net_struc}/tpr-fpr.png", dpi=400, bbox_inches='tight')
    #     plt.show()
    #
    #     # 绘制PR曲线
    #     plt.figure()
    #     plt.plot([0, 1], [1, 0], 'k--', linewidth=1)
    #     plt.plot(recall, precision, 'b-', label=f"Bayes(AUC={metrics.auc(recall, precision):.3f})", linewidth=1)
    #     plt.legend(loc="best")
    #     plt.xlabel("Recall")
    #     plt.ylabel("Precision")
    #     plt.title(f"Bayes PR Curve")
    #     plt.savefig(fig_par_dir + f"{network.net_struc}/precision-recall.png", dpi=400, bbox_inches='tight')
    #     plt.show()

    styles = ['b-', 'r:', 'm--', 'g-.']
    # 设置画布
    fig1, ax1 = plt.subplots()
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax1.set_xlabel("FPR")
    ax1.set_ylabel("TPR")
    ax1.set_title(f"Bayes ROC Curve")
    axins1 = ax1.inset_axes([0.3, 0.3, 0.3, 0.3])
    fig2, ax2 = plt.subplots()
    ax2.plot([0, 1], [1, 0], 'k--', linewidth=1)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title(f"Bayes PR Curve")

    for network, style in zip(donor_networks, styles):
        # if network.net_struc != "hillclimb":
        #     continue
        scores_testing = network.predict_scores(signals_testing)
        fpr, tpr, thr = metrics.roc_curve(labels_testing, scores_testing)
        precision, recall, thresholds = metrics.precision_recall_curve(labels_testing, scores_testing)

        # 绘制ROC曲线
        ax1.plot(fpr, tpr, style,
                 label=f"{network.net_struc.capitalize()} (AUC={metrics.auc(fpr, tpr):.3f})",
                 linewidth=0.8)
        axins1.plot(fpr, tpr, style)
        # 绘制PR曲线
        ax2.plot(recall, precision, style,
                 label=f"{network.net_struc.capitalize()} (AUC={metrics.auc(recall, precision):.3f})",
                  linewidth=0.8)

    axins1.set_xlim(0, 0.1)
    axins1.set_ylim(0.85, 0.95)
    mark_inset(ax1, axins1, loc1=3, loc2=1)
    ax1.legend(loc="lower right")
    ax2.legend(loc="lower left")
    fig1.savefig(fig_par_dir + f"tpr-fpr.png", dpi=400, bbox_inches='tight')
    fig2.savefig(fig_par_dir + f"precision-recall.png", dpi=400, bbox_inches='tight')
    plt.show()
    
    """************************************基于结构学习的贝叶斯网络************************************"""
    # # 计算f1-score最大的阈值
    # scores_testing = donor_hillclimb_network.predict_scores(signals_testing)
    # precision, recall, thresholds = metrics.precision_recall_curve(labels_testing, scores_testing)
    # max_f1score = 0
    # ulti_thr = 0.5
    # thrs = np.arange(min(thresholds), max(thresholds), 0.05)
    # for thr in thrs:
    #     labels_pred = np.zeros(len(signals_testing), dtype=int)
    #     labels_pred[scores_testing >= thr] = 1
    #     f1score = metrics.f1_score(labels_testing, labels_pred)
    #     if f1score > max_f1score:
    #         max_f1score = f1score
    #         ulti_thr = thr
    # print(max_f1score, ulti_thr)

    end = datetime.datetime.now()
    print(f"程序运行时间： {end - start}")
    # sys.exit(0)
