# -*- coding: utf-8 -*-
# @Time    : 2022/7/10 9:38
# @Author  : Terence Tan
# @Email   : 2228254095@qq.com
# @FileName: Predictor_GUI.py
# @Software: PyCharm

import sys

from PyQt5 import QtGui

sys.path.append("./Source")

from PyQt5.QtWidgets import (QWidget, QPushButton, QTextEdit, QLineEdit, QFileDialog,
                             QApplication, QMessageBox, QLabel, QGridLayout)
from PyQt5.QtGui import QIcon, QFont
from Source.gene_bayes import BN
from Source.signals_processing import check_signal


class Predictor(QWidget):
    def __init__(self):
        super().__init__()

        # 加载模型
        self.du_len, self.dd_len = 3, 6
        self.donor_model_dir = f"./Models/Bayes/donor/u3d6_u15d2/"
        self.donor_clf = BN(self.du_len, self.dd_len, self.donor_model_dir, "donor")
        self.donor_clf.load_params()
        self.donor_thr = 3.02
        self.au_len, self.ad_len = 25, 9
        self.acceptor_model_dir = f"./Models/Bayes/acceptor/u9d15_u25d9/"
        self.acceptor_clf = BN(self.au_len, self.ad_len, self.acceptor_model_dir, "acceptor")
        self.acceptor_clf.load_params()
        self.acceptor_thr = 3.87

        # 上传文件或输入序列
        self.btn_infile = QPushButton("Browse...", self)
        self.btn_infile.setFont(QFont("微软雅黑", 10))
        self.btn_infile.setToolTip("Sequence must be in FASTA format or pure sequence")
        self.btn_infile.clicked.connect(self.get_seq)
        self.seq = QTextEdit(self)
        self.seq.setFont(QFont("Century Schoolbook", 10))
        self.file_status = QLineEdit(self)
        self.file_status.setText("No file selected")
        self.file_status.setFont(QFont("宋体", 10))

        # 提交、重置、下载
        self.btn_submit = QPushButton("Submit", self)
        self.btn_submit.setFont(QFont("微软雅黑", 10))
        self.btn_submit.clicked.connect(self.predict)
        self.btn_reset = QPushButton("Reset", self)
        self.btn_reset.setFont(QFont("微软雅黑", 10))
        self.btn_reset.clicked.connect(self.reset)
        self.btn_download = QPushButton(QIcon("./Figures/download.ico"), "Download", self)
        self.btn_download.setFont(QFont("微软雅黑", 10))
        self.btn_download.clicked.connect(self.download)

        # 预测结果
        self.donor_sites = QTextEdit(self)
        self.donor_sites.setFont(QFont("Ebrima", 10))
        self.acceptor_sites = QTextEdit(self)
        self.acceptor_sites.setFont(QFont("Ebrima", 10))

        lbl1 = QLabel("Upload your sequences")
        lbl1.setFont(QFont("Arial", 10))
        lbl2 = QLabel("or you can enter your sequences:")
        lbl2.setFont(QFont("Arial", 10))
        lbl3 = QLabel("Donor splicing sites")
        lbl3.setFont(QFont("Arial", 10))
        lbl4 = QLabel("Acceptor splicing sites")
        lbl4.setFont(QFont("Arial", 10))

        grid = QGridLayout()
        grid.setSpacing(10)
        row = 1
        grid.addWidget(lbl1, row, 0, 1, 1)
        row += 1
        grid.addWidget(self.btn_infile, row, 0, 1, 1)
        grid.addWidget(self.file_status, row, 1, 1, 2)
        row += 1
        grid.addWidget(lbl2, row, 0, 1, 1)
        row += 1
        grid.addWidget(self.seq, row, 0, 1, 3)
        row += 1
        grid.addWidget(self.btn_submit, row, 0, 1, 1)
        grid.addWidget(self.btn_reset, row, 1, 1, 1)
        grid.addWidget(self.btn_download, row, 2, 1, 1)
        row += 1
        grid.addWidget(lbl3, row, 0, 1, 1)
        row += 1
        grid.addWidget(self.donor_sites, row, 0, 1, 3)
        row += 1
        grid.addWidget(lbl4, row, 0, 1, 1)
        row += 1
        grid.addWidget(self.acceptor_sites, row, 0, 1, 3)

        self.setLayout(grid)

        self.setGeometry(500, 300, 500, 600)
        self.setWindowTitle("A Splicing Site Prediction Tool")
        self.setWindowIcon(QIcon("./Figures/icon.ico"))
        self.show()

    def get_seq(self):
        openfile_name = QFileDialog.getOpenFileName(self, "Select file", "", "所有文件(*.*)")
        try:
            file_path = openfile_name[0]
            self.file_status.setText(file_path.split("/")[-1])
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.seq.setPlainText(content)
        except:
            pass

    def check_signal_new(self, up_len, down_len, signal, site_type):
        """
        检查信号
        """
        signal_len = up_len + down_len
        if site_type == "donor":
            if signal[up_len: up_len + 2] == "gt" and check_signal(signal, signal_len):
                return True
        else:
            if signal[up_len - 2: up_len] == "ag" and check_signal(signal, signal_len):
                return True
        return False

    def predict_sites(self, up_len, down_len, clf, thr, site_type):
        """
        位点预测
        """
        content = self.seq.toPlainText()
        seq = "".join([row.strip() for row in content.split() if not row.startswith('>')])
        signal_len = up_len + down_len
        signals = []
        sites = []

        for idx in range(len(seq) - signal_len + 1):
            slide = seq[idx: idx + signal_len].lower()
            if self.check_signal_new(up_len, down_len, slide, site_type):
                signals.append(slide)
                sites.append(idx)

        labels = clf.predict(signals, thr)
        for idx in range(labels.shape[0]):
            if labels[idx] == 1:
                if site_type == "donor":
                    self.donor_sites.insertPlainText(f"{sites[idx] + up_len}\n")
                else:
                    self.acceptor_sites.insertPlainText(f"{sites[idx] + up_len + 1}\n")

    def predict(self):
        if not self.seq.toPlainText():
            QMessageBox.information(self, "Hint", "Please enter the sequence first")
            return
        self.donor_sites.setPlainText("")
        self.acceptor_sites.setPlainText("")
        self.predict_sites(self.du_len, self.dd_len, self.donor_clf, self.donor_thr, "donor")
        self.predict_sites(self.au_len, self.ad_len, self.acceptor_clf, self.acceptor_thr, "acceptor")

    def reset(self):
        self.seq.setPlainText("")
        self.file_status.setText("No file selected")
        self.donor_sites.setPlainText("")
        self.acceptor_sites.setPlainText("")

    def download(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select folder Path", './')
        with open(folder_path + "/result.txt", "w") as f:
            f.write("Donor splicing sites:\n")
            f.write(self.donor_sites.toPlainText())
            f.write("\nAcceptor splicing sites:\n")
            f.write(self.acceptor_sites.toPlainText())

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message',
                                     "Are you sure to quit?", QMessageBox.Yes |
                                     QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Predictor()
    sys.exit(app.exec_())
