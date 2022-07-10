# SplicePred
基于权重阵列模型(WAM)、贝叶斯网络(BN)、支持向量机(SVM)的剪接位点识别.

## Abstract
剪接位点(Splicing site)的识别作为基因识别中的一个重要环节一直受到研究人员的关注，考虑到剪接位点附近存在的序列保守性，已有一些基于统计特性的方法被用于剪接位点的识别。
文中将权重阵列模型(Weight Array Model, WAM)、贝叶斯网络(Bayesian Network, BN)和支持向量机(Support Vector Machine, SVM)三种方法用于剪接位点的识别中，并使用受试者操作特征(Receiver Operating Characteristic, ROC)曲线和精度-召回率(Precision Recall, PR)曲线及其曲线下面积(Area Under the Curve, AUC)度量模型识别效果。首先，通过WMM和WAM模型的比较测试，表明纳入碱基之间的依赖性信息可以有效提升识别效果。其次，采用基于贝叶斯网络建模的方法，设计了不同网络结构，包括朴素贝叶斯模型、链式贝叶斯网络、扩展贝叶斯网络以及基于结构学习的贝叶斯网络结构，其中基于结构学习的贝叶斯网络具有最好的识别效果。最后，应用支持向量机进行剪接供体位点识别，并分别采用了稀疏编码(Sparse encoding)、一阶马尔科夫编码(MM1 encoding)、二阶马尔科夫编码(MM2 encoding)和FDTF encoding的编码方法，将编码特征用于SVM识别，不过后三种相对复杂的编码方法并未能在识别效果上对比简单的稀疏编码方法展现出优势。

## Requirements
Please see [requirements.txt](requirements.txt)

## Models
Weighted Array Model, Bayesian network, and Support vector machine.

See [Source](Source) for details in implementation.
