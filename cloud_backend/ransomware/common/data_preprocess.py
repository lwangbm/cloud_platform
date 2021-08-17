from __future__ import division

import os
import pickle

import numpy as np
import tensorflow as tf
from scipy import stats
from scipy.fftpack import fft
from sklearn import preprocessing

from utils.constants import (CLEAN_DATA_TEST, CLEAN_DATA_TRAIN, NEGATIVE_LABEL,
                             NEGATIVE_SUBDIR, POSITIVE_LABEL, POSITIVE_SUBDIR,
                             SAMPLING_FREQ)

tf.compat.v1.app.flags.DEFINE_bool("print_log", True, "whether to print temp results")
tf.compat.v1.app.flags.DEFINE_string(
    "train_dir", "data/training", "Root path of training data"
)
tf.compat.v1.app.flags.DEFINE_string(
    "test_dir", "data/testing", "Root path of testing data"
)

FLAGS = tf.compat.v1.app.flags.FLAGS

# path parse
training_positive_dir = os.path.join(FLAGS.train_dir, POSITIVE_SUBDIR)  # 正样本目录
training_negative_dir = os.path.join(FLAGS.train_dir, NEGATIVE_SUBDIR)  # 负样本目录
testing_positive_dir = os.path.join(FLAGS.test_dir, POSITIVE_SUBDIR)  # 正样本目录
testing_negative_dir = os.path.join(FLAGS.test_dir, NEGATIVE_SUBDIR)  # 负样本目录
# test_dir = os.path.join(FLAGS.train_dir, TEST_SUBDIR)  # 待测样本目录

# logs parse
training_positive_logs, training_negative_logs = os.listdir(
    training_positive_dir
), os.listdir(training_negative_dir)
testing_positive_logs, testing_negative_logs = os.listdir(
    testing_positive_dir
), os.listdir(testing_negative_dir)


def training_data_preproc():
    feature_positive, feature_negative, label_positive, label_negative = [], [], [], []
    # 计算positive样本特征值
    for filename in list(set(training_positive_logs)):
        # positive_logs.sort(key=lambda x: int(x[:-5]))
        if ".txt" in filename:
            # 计算feature并贴标签：‘1’：label
            print("calculating positive feature：%s" % filename)
            feature_positive.extend(
                feature_extra(os.path.join(training_positive_dir, filename))
            )
            label_positive = [POSITIVE_LABEL] * len(feature_positive)
    # 计算negative样本特征值
    for filename in list(set(training_negative_logs)):
        if ".txt" in filename:
            # 计算feature并贴标签：‘2’：label
            print("calculating negative feature：%s" % filename)
            feature_negative.extend(
                feature_extra(os.path.join(training_negative_dir, filename))
            )
            label_negative = [NEGATIVE_LABEL] * len(feature_negative)

    x = np.concatenate((feature_positive, feature_negative), axis=0)
    y = np.concatenate((label_positive, label_negative), axis=0)
    scaler = preprocessing.StandardScaler()
    x = scaler.fit_transform(x)
    # rm avg and median
    x_original = x
    x = x[:, :-2]
    np.savez(CLEAN_DATA_TRAIN, x=x, y=y, x_original=x_original)

    with open("checkpoint/scaler.pickle", "wb") as f:
        pickle.dump(scaler, f)
        print("scaler save successfully")


def testing_data_preproc(scaler):

    feature_positive, feature_negative, label_positive, label_negative = [], [], [], []
    # 计算positive样本特征值
    for filename in list(set(testing_positive_logs)):
        # positive_logs.sort(key=lambda x: int(x[:-5]))
        if ".txt" in filename:
            # 计算feature并贴标签：‘1’：label
            print("calculating positive feature：%s" % filename)
            feature_positive.extend(
                feature_extra(os.path.join(testing_positive_dir, filename))
            )
            label_positive = [POSITIVE_LABEL] * len(feature_positive)

    # 计算negative样本特征值
    for filename in list(set(testing_negative_logs)):
        if ".txt" in filename:
            # 计算feature并贴标签：‘2’：label
            print("calculating negative feature：%s" % filename)
            feature_negative.extend(
                feature_extra(os.path.join(testing_negative_dir, filename))
            )
            label_negative = [NEGATIVE_LABEL] * len(feature_negative)

    x = np.concatenate((feature_positive, feature_negative), axis=0)
    y = np.concatenate((label_positive, label_negative), axis=0)
    x = scaler.transform(x)
    # rm avg and median
    x_original = x
    x = x[:, :-2]
    np.savez(CLEAN_DATA_TEST, x=x, y=y, x_original=x_original)


def calculate_stat(data):
    # 集中趋势的度量
    avg = np.mean(data)  # 计算平均值
    median = np.median(data)  # 中位数
    up4 = np.quantile(data, 0.25)
    down4 = np.quantile(data, 0.75)
    xc = up4  # 上四分位数
    xd = down4  # 下四分位数
    # 离散趋势的度量
    ya = max(data)  # 最大值
    yb = min(data)  # 最小值
    yc = max(data) - min(data)  # 极差
    yd = down4 - up4  # 四分位距
    std = np.std(data)  # 计算标准差 ***
    yf = np.var(data)  # 计算方差
    yg = np.std(data) / np.mean(data)  # 离散系数
    # 偏度与峰度的度量
    za = stats.skew(data)  # 计算偏斜度
    zb = stats.kurtosis(data)  # 计算峰度
    w_fft = fft(data)
    w_half = w_fft[range(int(len(w_fft) / 2))]  # 由于对称性，只取一半区间
    w_abs = abs(w_half)
    w_f = np.arange(len(data))
    wa = max(w_abs)
    wb = min(w_abs)
    return np.array(
        [xc, xd, ya, yb, yc, yd, yf, yg, za, zb, wa, wb, std, avg, median]
    ).reshape(1, -1)


# 提取数据，并得到特征值
def feature_extra(file):
    # 提取数据并保存在列表中
    feature = np.zeros([0, 15], float)
    raw_data = []
    with open(file, "r", encoding="UTF-8") as file_in:
        for y in file_in.read().split("\n"):
            try:
                tmp = float(y)
            except:
                continue
            raw_data.append(tmp)
    # 将数据分块————分块的长度和数据采样频率共同决定
    chunk_data = [
        raw_data[x : x + SAMPLING_FREQ] for x in range(0, len(raw_data), SAMPLING_FREQ)
    ]
    # 提取特征值
    for k in range(0, len(chunk_data)):
        tmp = calculate_stat(chunk_data[k])
        if not np.isnan(tmp).any():
            feature = np.append(feature, tmp, 0)

    return feature
