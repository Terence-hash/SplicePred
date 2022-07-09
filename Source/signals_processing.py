# -*- coding: utf-8 -*-
# @Time    : 2022/4/19 16:00
# @Author  : Terence Tan
# @Email   : 2228254095@qq.com
# @FileName: signals_processing.py
# @Software: PyCharm

import datetime
import os
import re

bases = {'a': 0, 'g': 1, 'c': 2, 't': 3}


def check_signal(signal, signal_len):
    """
    检查信号
    """
    # 排除长度异常信号
    if len(signal) != signal_len:
        return False
    # 排除非常见碱基
    for base in signal:
        if base not in bases.keys():
            return False
    return True


def extract_target_signals(whole_seq, sites, signal_ulen, signal_dlen, site_type):
    """
    提取目标信号
    """
    target_signals = []
    signal_len = signal_ulen + signal_dlen
    for site in sites:
        if site_type == "donor":
            signal = whole_seq[site - signal_ulen: site + signal_dlen].lower()
        else:
            signal = whole_seq[site - signal_ulen - 1: site + signal_dlen - 1].lower()
        if check_signal(signal, signal_len):
            target_signals.append(signal)

    return target_signals


def extract_bgd_signals(whole_seq, signal_ulen, signal_dlen, site_type, target_signals=None, excluding=True):
    """
    提取背景信号
    """
    if target_signals is None:
        target_signals = []
    bgd_signals = []
    signal_len = signal_ulen + signal_dlen
    for i in range(len(whole_seq) - signal_len + 1):
        slide = whole_seq[i: i + signal_len].lower()
        if slide in target_signals:
            continue
        if site_type == "donor":
            # 对于donor，排除非GT中心背景信号
            if excluding and slide[signal_ulen: signal_ulen + 2] != "gt":
                continue
        else:
            # 对于acceptor，排除非AG中心背景信号
            if excluding and slide[signal_ulen - 2: signal_ulen] != "ag":
                continue
        if check_signal(slide, signal_len):
            bgd_signals.append(slide)

    return bgd_signals


def extract_signals(dir_path, donor_ulen, donor_dlen, acceptor_ulen, acceptor_dlen, excluding=True):
    """
    提取剪接位点信号并保存
    :param dir_path: 文件目录
    :param donor_ulen: donor位点的外显子-内含子分界的上游长度
    :param donor_dlen: donor位点的外显子-内含子分界的下游长度
    :param acceptor_ulen: acceptor位点的外显子-内含子分界的上游长度
    :param acceptor_dlen: acceptor位点的外显子-内含子分界的下游长度
    :param excluding: True,表示排除非GT(AG)中心序列，否则保留
    :return: None
    """
    donor_signals, acceptor_signals = [], []
    bd_signals, ba_signals = [], []
    exon_num, intron_num = 0, 0

    for file_name in os.listdir(dir_path):
        pattern = r"\((.*?)\)"
        with open(dir_path + file_name, 'r') as f:
            content = f.readlines()
            whole_seq = "".join([row.strip() for row in content[2:]])
            exons = re.findall(pattern, content[1])[0].split(',')
            intron_num += len(exons) - 1
            exon_num += len(exons)
            donor_sites = [int(site.split("..")[1]) for site in exons[:-1]]
            acceptor_sites = [int(site.split("..")[0]) for site in exons[1:]]

            # 提取donor和acceptor目标信号
            donor_signals.extend(extract_target_signals(whole_seq, donor_sites, donor_ulen,
                                                        donor_dlen, site_type="donor"))
            acceptor_signals.extend(extract_target_signals(whole_seq, acceptor_sites, acceptor_ulen,
                                                           acceptor_dlen, site_type="acceptor"))
            # 提取donor和acceptor对应背景信号
            bd_signals.extend(extract_bgd_signals(whole_seq, donor_ulen, donor_dlen,
                                                  site_type="donor",
                                                  target_signals=donor_signals,
                                                  excluding=True))
            ba_signals.extend(extract_bgd_signals(whole_seq, acceptor_ulen, acceptor_dlen,
                                                  site_type="acceptor",
                                                  target_signals=acceptor_signals,
                                                  excluding=True))

    dataset = dir_path.split("/")[-2].split()[0].lower()
    print(f"    外显子数量：{exon_num}，内含子数量：{intron_num}")
    print(f"    donor信号数量：{len(donor_signals)}，donor背景信号数量：{len(bd_signals)}")
    print(f"    acceptor信号数量：{len(acceptor_signals)}，acceptor背景信号数量：{len(ba_signals)}")

    # 保存4种信号数据集
    signals_dir = f"../Data_files/feature_data/u{donor_ulen}d{donor_dlen}_u{acceptor_ulen}d{acceptor_dlen}/"
    if not os.path.exists(signals_dir):
        os.mkdir(signals_dir)
    with open(signals_dir + f"donor_signals_{dataset}.txt", 'w') as df, \
            open(signals_dir + f"acceptor_signals_{dataset}.txt", 'w') as af, \
            open(signals_dir + f"bd_signals_{dataset}.txt", 'w') as bdf, \
            open(signals_dir + f"ba_signals_{dataset}.txt", 'w') as baf:
        for signal in donor_signals:
            df.writelines([signal, '\n'])
        for signal in acceptor_signals:
            af.writelines([signal, '\n'])
        for signal in bd_signals:
            bdf.writelines([signal, '\n'])
        for signal in ba_signals:
            baf.writelines([signal, '\n'])


def load_signals(feat_dir, site_type, dataset):
    """
    加载信号
    """
    with open(feat_dir + f"{site_type}_signals_{dataset}.txt", 'r') as pf, \
            open(feat_dir + f"b{site_type[0]}_signals_{dataset}.txt", 'r') as nf:
        target_signals = [i.strip() for i in pf.readlines()]
        bg_signals = [i.strip() for i in nf.readlines()]

    return target_signals, bg_signals


if __name__ == '__main__':
    start = datetime.datetime.now()
    # 数据路径
    training_path = "../Data_files/raw_data/Training set/"
    testing_path = "../Data_files/raw_data/Testing set/"
    # 提取核苷酸序列信号
    for donor_ulen, donor_dlen, acceptor_ulen, acceptor_dlen in ([6, 9, 15, 6], [9, 12, 20, 9], [9, 15, 27, 9]):
        extract_signals(training_path, donor_ulen, donor_dlen, acceptor_ulen, acceptor_dlen)
        extract_signals(testing_path, donor_ulen, donor_dlen, acceptor_ulen, acceptor_dlen)

    # 输出运行时间
    end = datetime.datetime.now()
    print(f"程序运行时间： {end - start}")
