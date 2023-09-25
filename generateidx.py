# Load data and IP clustering

import math
import random
import pandas as pd
import numpy as np
import argparse
from sklearn import preprocessing
from utils import MaxMinScaler
from tqdm import tqdm
import csv

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='Shanghai', choices=["Shanghai", "New_York", "Los_Angeles"],
                    help='which dataset to use')
parser.add_argument('--train_test_ratio', type=float, default=0.8, help='landmark ratio')
parser.add_argument('--lm_ratio', type=float, default=0.7, help='landmark ratio')
parser.add_argument('--seed', type=int, default=1234)
# parser.add_argument('--seed', type=int, default=2022)


opt = parser.parse_args()
print("Dataset: ", opt.dataset)


def get_XY(dataset):
    data_path = "./datasets/{}/data.csv".format(dataset)
    ip_path = './datasets/{}/ip.csv'.format(dataset)
    trace_path = './datasets/{}/last_traceroute.csv'.format(dataset)

    data_origin = pd.read_csv(data_path, encoding='gbk', low_memory=False)
    ip_origin = pd.read_csv(ip_path, encoding='gbk', low_memory=False)
    trace_origin = pd.read_csv(trace_path, encoding='gbk', low_memory=False)

    data = pd.concat([data_origin, ip_origin, trace_origin], axis=1)
    data.fillna({"isp": '0'}, inplace=True)

    # labels
    Y = data[['longitude', 'latitude']]
    Y = np.array(Y)

    # features
    if dataset == "Shanghai":  # Shanghai，27+8+16, 共51维，其中8+16=24维为traceroute相关measurment
        # classification features
        X_class = data[['orgname', 'asname', 'address', 'isp']]
        scaler = preprocessing.OneHotEncoder(sparse=False)
        X_class = scaler.fit_transform(X_class)

        X_class1 = data['isp']
        X_class1 = preprocessing.LabelEncoder().fit_transform(X_class1)
        X_class1 = preprocessing.MinMaxScaler().fit_transform(np.array(X_class1).reshape((-1, 1)))

        X_2 = data[['ip_split1', 'ip_split2', 'ip_split3', 'ip_split4']]
        X_2 = preprocessing.MinMaxScaler().fit_transform(np.array(X_2))

        X_3 = data['asnumber']
        X_3 = preprocessing.LabelEncoder().fit_transform(X_3)
        X_3 = preprocessing.MinMaxScaler().fit_transform(np.array(X_3).reshape(-1, 1))

        X_4 = data[['aiwen_ping_delay_time', 'vp806_ping_delay_time', 'vp808_ping_delay_time', 'vp813_ping_delay_time']]
        delay_scaler = MaxMinScaler()
        delay_scaler.fit(X_4)
        X_4 = delay_scaler.transform(X_4)

        X_5 = data[['aiwen_tr_steps', 'vp806_tr_steps', 'vp808_tr_steps', 'vp813_tr_steps']]
        step_scaler = MaxMinScaler()
        step_scaler.fit(X_5)
        X_5 = step_scaler.transform(X_5)

        X_6 = data[
            ['aiwen_last1_delay', 'aiwen_last2_delay_total', 'aiwen_last3_delay_total', 'aiwen_last4_delay_total',
             'vp806_last1_delay', 'vp806_last2_delay_total', 'vp806_last3_delay_total', 'vp806_last4_delay_total',
             'vp808_last1_delay', 'vp808_last2_delay_total', 'vp808_last3_delay_total', 'vp808_last4_delay_total',
             'vp813_last1_delay', 'vp813_last2_delay_total', 'vp813_last3_delay_total', 'vp813_last4_delay_total']]
        X_6 = np.array(X_6)
        X_6[X_6 <= 0] = 0
        X_6 = preprocessing.MinMaxScaler().fit_transform(X_6)

        X = np.concatenate([X_class1, X_class, X_2, X_3, X_4, X_5, X_6], axis=1)
        # without isp
        # X = np.concatenate([X_class, X_2, X_3, X_4, X_5, X_6], axis=1)

    elif dataset == "New_York" or "Los_Angeles":  # New_York or Los_Angeles, 6+8+16, 共30维, 其中8+16=24维为tracerout相关measurment
        X_class = data['isp']
        X_class = preprocessing.LabelEncoder().fit_transform(X_class)
        X_class = preprocessing.MinMaxScaler().fit_transform(np.array(X_class).reshape((-1, 1)))

        X_2 = data[['ip_split1', 'ip_split2', 'ip_split3', 'ip_split4']]
        X_2 = preprocessing.MinMaxScaler().fit_transform(np.array(X_2))

        X_3 = data['as_mult_info']
        X_3 = preprocessing.LabelEncoder().fit_transform(X_3)
        X_3 = preprocessing.MinMaxScaler().fit_transform(np.array(X_3).reshape(-1, 1))

        X_4 = data[['vp900_ping_delay_time', 'vp901_ping_delay_time', 'vp902_ping_delay_time', 'vp903_ping_delay_time']]
        delay_scaler = MaxMinScaler()
        delay_scaler.fit(X_4)
        X_4 = delay_scaler.transform(X_4)

        X_5 = data[['vp900_tr_steps', 'vp901_tr_steps', 'vp902_tr_steps', 'vp903_tr_steps']]
        step_scaler = MaxMinScaler()
        step_scaler.fit(X_5)
        X_5 = step_scaler.transform(X_5)

        X_6 = data[
            ['vp900_last1_delay', 'vp900_last2_delay_total', 'vp900_last3_delay_total', 'vp900_last4_delay_total',
             'vp901_last1_delay', 'vp901_last2_delay_total', 'vp901_last3_delay_total', 'vp901_last4_delay_total',
             'vp902_last1_delay', 'vp902_last2_delay_total', 'vp902_last3_delay_total', 'vp902_last4_delay_total',
             'vp903_last1_delay', 'vp903_last2_delay_total', 'vp903_last3_delay_total', 'vp903_last4_delay_total']]
        X_6 = np.array(X_6)
        X_6[X_6 <= 0] = 0
        X_6 = preprocessing.MinMaxScaler().fit_transform(X_6)

        X = np.concatenate([X_2, X_class, X_3, X_4, X_5, X_6], axis=1)
        # without isp
        # X = np.concatenate([X_2, X_3, X_4, X_5, X_6], axis=1)

    return X, Y, np.array(trace_origin)


def get_cols(row, mode="odd"):
    start = 0 if mode == "odd" else 1
    idxs = range(start, row.size, 2)
    list = []
    for i in idxs:
        list.append(row[i])
    return np.array(list)


def find_all_nearest_router(row):
    last_router_idx = list(range(0, 32, 8))
    last_delay_idx = list(range(1, 32, 8))
    routers = row[last_router_idx]
    delays = row[last_delay_idx]
    # idx = [i for i in range(4) if delays[i]>0]

    return routers, delays


def find_nearest_router_lm(row):
    last_router_idx = list(range(0, 32, 8))
    last_delay_idx = list(range(1, 32, 8))
    routers = row[last_router_idx]
    delays = row[last_delay_idx]
    delays[delays <= 0] = math.inf
    nearest_idx = np.argmin(delays)

    # routers[routers == "-1"] = "-9999"

    return routers[nearest_idx], delays[nearest_idx]


def find_nearest_router_tg(row):
    last_router_idx = list(range(0, 32, 8))
    last_delay_idx = list(range(1, 32, 8))
    routers = row[last_router_idx]
    delays = row[last_delay_idx]
    delays[delays <= 0] = math.inf
    nearest_idx = np.argmin(delays)

    routers[routers == "-1"] = "-9999"

    return routers[nearest_idx], delays[nearest_idx]



def handle_common(common_router, landmarks, targets):
    data = {
        "router": common_router,
        "exist": False
    }
    if common_router == "-1":
        return data
    lm_idx = np.argwhere(landmarks["router"] == common_router)
    tg_idx = np.argwhere(targets["router"] == common_router)
    if len(tg_idx) < 1:
        return data
    lm_nodes = landmarks["X"][lm_idx]
    lm_labels = landmarks["Y"][lm_idx]
    lm_delays = landmarks["delay"][lm_idx]
    tg_nodes = targets["X"][tg_idx]
    tg_labels = targets["Y"][tg_idx]
    tg_delays = targets["delay"][tg_idx]
    center = lm_labels.mean(axis=0)
    data = {
        "lm_X": lm_nodes,
        "lm_Y": lm_labels,
        "lm_delay": lm_delays,
        "tg_X": tg_nodes,
        "tg_Y": tg_labels,
        "tg_delay": tg_delays,
        "center": center,
        "router": common_router,
        "exist": True
    }
    return data


def get_idx(num, seed, train_test_ratio, lm_ratio):
    idx = list(range(0, num))
    random.seed(seed)
    random.shuffle(idx)
    lm_train_num = int(num * train_test_ratio * lm_ratio)
    tg_train_num = int(num * train_test_ratio * (1 - lm_ratio))

    lm_train_idx, tg_train_idx, tg_test_idx = idx[:lm_train_num], \
                                              idx[lm_train_num:tg_train_num + lm_train_num], \
                                              idx[lm_train_num + tg_train_num:]
    return lm_train_idx, tg_train_idx, lm_train_idx + tg_train_idx, tg_test_idx


def get_graph(dataset, lm_idx, tg_idx, mode):
    X, Y, T = get_XY(dataset)  # preprocess whole dataset

    last_hop_tg = list(map(find_nearest_router_tg, T))  # [(ip, time delay),...]
    last_routers_tg = np.array(last_hop_tg)[:, 0]

    last_hop_lm = list(map(find_nearest_router_lm, T))  # [(ip, time delay),...]
    last_routers_lm = np.array(last_hop_lm)[:, 0]

    has_neighbor_targets1 = []
    has_neighbor_targets10 = []
    for id in tqdm(tg_idx):
        neighbors = set()
        router = last_routers_tg[id]
        for j in lm_idx:
            if last_routers_lm[j] == router:
                neighbors.add(j)
        if 0 < len(neighbors) <= 10:  # 若target存在neighbors
            has_neighbor_targets1.append(id)
        elif 10 < len(neighbors):
            has_neighbor_targets10.append(id)

    return has_neighbor_targets1, has_neighbor_targets10


if __name__ == '__main__':
    seed = opt.seed

    train_test_ratio = opt.train_test_ratio  # 0.8
    lm_ratio = opt.lm_ratio  # 0.7
    lm_train_idx, tg_train_idx, lm_test_idx, tg_test_idx = get_idx(len(get_XY(opt.dataset)[0]), seed,
                                                                   train_test_ratio,
                                                                   lm_ratio)  # split train and test

    print("loading train set...")
    train_targets1, train_targets10 = get_graph(opt.dataset, lm_train_idx, tg_train_idx, mode="train")
    print("train set loaded.")

    print("loading test set...")
    test_targets1, test_targets10 = get_graph(opt.dataset, lm_test_idx, tg_test_idx, mode="test")
    print("test set loaded.")
    print()

    np.savez("datasets/{}/target_idx_lm{}.npz".format(opt.dataset, seed), train_tg_idx1=train_targets1,
             train_tg_idx10=train_targets10,test_tg_idx1=test_targets1, test_tg_idx10 = test_targets10)

    print("finish!")
