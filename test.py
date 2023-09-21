# -*- coding: utf-8 -*-

"""
    load checkpoint and then test
"""

import torch.nn

from  utils  import *
import argparse
import numpy as np
import random
from model_test import *
import copy
import pandas as pd

parser = argparse.ArgumentParser()
# parameters of training
parser.add_argument("--beta", type=float, default=0.01, help="Default is 1e-5")
parser.add_argument("--beta_x", type=float, default=10, help="Default is 1e-5")
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)

parser.add_argument('--lr', type=float, default=2e-3)
parser.add_argument('--harved_epoch', type=int, default=5)
parser.add_argument('--early_stop_epoch', type=int, default=50)
parser.add_argument('--saved_epoch', type=int, default=100)
parser.add_argument('--load_epoch', type=int, default=8)
parser.add_argument('--dropout', type=float, default=0.1)
# parameters of initializing
parser.add_argument('--seed', type=int, default=1234, help='manual seed')
parser.add_argument('--model_name', type=str, default='RIPGeo')
parser.add_argument('--dataset', type=str, default='Shanghai', choices=["Shanghai", "New_York", "Los_Angeles"],
                    help='which dataset to use')
# parameters of model
parser.add_argument('--dim_in', type=int, default=30, choices=[51, 30], help="51 if Shanghai / 30 else")
parser.add_argument('--dim_med', type=int, default=32)
parser.add_argument('--dim_z', type=int, default=32)
parser.add_argument('--eta', type=float, default=0.1)
parser.add_argument('--zeta', type=float, default=0.1)
parser.add_argument('--step', type=int, default=2)
parser.add_argument('--mu', type=float, default=0.2)
parser.add_argument('--lambda_1', type=float, default=1)
parser.add_argument('--lambda_2', type=float, default=1)
parser.add_argument('--c_mlp', type=bool, default=True)
parser.add_argument('--epoch_threshold', type=int, default=50)


opt = parser.parse_args()
print("Learning rate: ", opt.lr)
print("Dataset: ", opt.dataset)

if opt.seed:
    print("Random Seed: ", opt.seed)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
torch.set_printoptions(threshold=float('inf'))

warnings.filterwarnings('ignore')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device:", device)
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

'''load data'''
train_data = np.load("./datasets/{}/Clustering_s1234_graph70_train.npz".format(opt.dataset),
                     allow_pickle=True)
test_data = np.load("./datasets/{}/Clustering_s1234_graph70_test.npz".format(opt.dataset),
                    allow_pickle=True)
train_data, test_data = train_data["data"], test_data["data"]
print("data loaded.")


if __name__ == '__main__':
    train_data, test_data = get_data_generator(opt, train_data, test_data, normal=2)

    reserve_ratios = np.arange(0.1, 1.05, 0.05)
    # reserve_ratios = [2, 5, 10, 30, 50]
    result_list = []
    for reserve_ratio in reserve_ratios:
        checkpoint = torch.load(f"asset/model/{opt.dataset}_{opt.load_epoch}.pth")
        print(f"Load model asset/model/{opt.dataset}_{opt.load_epoch}.pth")
        model = eval("Geo")(dim_in=opt.dim_in, dim_z=opt.dim_z, dim_med=opt.dim_med, dim_out=2, collaborative_mlp=opt.c_mlp)
        model.load_state_dict(checkpoint['model_state_dict'])
        if cuda:
            model.cuda()

        # test
        total_mse, total_mae, test_num = 0, 0, 0
        dislist = []

        model.eval()
        distance_all = []
        with torch.no_grad():
            for i in range(len(test_data)):

                lm_X, lm_Y, tg_X, tg_Y, y_max, y_min = test_data[i]["lm_X"], test_data[i]["lm_Y"], \
                                                                            test_data[i]["tg_X"], test_data[i]["tg_Y"], \
                                                                            test_data[i]["y_max"], test_data[i]["y_min"]

                y_pred, _, _ = model(Tensor(lm_X), Tensor(lm_Y), Tensor(tg_X), Tensor(tg_Y), reserve_ratio)
                distance = dis_loss(Tensor(tg_Y), y_pred, y_max, y_min)
                for i in range(len(distance.cpu().detach().numpy())):
                    dislist.append(distance.cpu().detach().numpy()[i])
                    distance_all.append(distance.cpu().detach().numpy()[i])

                test_num += len(tg_Y)
                total_mse += (distance * distance).sum()
                total_mae += distance.sum()

            total_mse = total_mse / test_num
            total_mae = total_mae / test_num

            print(f"Reserve Ratio: {reserve_ratio:.2f}")
            print("test: mse: {:.4f}\trmse: {:.4f} \tmae: {:.4f}".format(total_mse, np.sqrt(total_mse.cpu().item()), total_mae))
            dislist_sorted = sorted(dislist)
            print('test median:', dislist_sorted[int(len(dislist_sorted) / 2)])


            reserve_ratio_formatted = "{:.2f}".format(reserve_ratio)
            total_mae_formatted = "{:.4f}".format(total_mae.item())
            result_list.append((reserve_ratio_formatted, total_mae_formatted))
    print(result_list)

    