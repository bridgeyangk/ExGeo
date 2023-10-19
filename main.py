# -*- coding: utf-8 -*-
import torch.nn
from utils import *
import argparse
import numpy as np
import random, os
from model import *
import copy
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
# from thop import profile



parser = argparse.ArgumentParser()
# parameters of initializing
parser.add_argument('--seed', type=int, default=1234, help='manual seed')
parser.add_argument('--model_name', type=str, default='ExGeo')
parser.add_argument('--dataset', type=str, default='Los_Angeles', choices=["Shanghai", "New_York", "Los_Angeles"],
                    help='which dataset to use')

# parameters of training
parser.add_argument("--beta", type=float, default=0.01, help="Default is 0.01")
parser.add_argument("--beta_x", type=float, default=0.1, help="Default is 1")
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)

parser.add_argument('--lr', type=float, default=2e-3)
parser.add_argument('--harved_epoch', type=int, default=5)
parser.add_argument('--early_stop_epoch', type=int, default=50)
parser.add_argument('--saved_epoch', type=int, default=5)
parser.add_argument('--load_epoch', type=int, default=100)

# parameters of model
parser.add_argument('--dim_in', type=int, default=51, choices=[51, 30], help="51 if Shanghai / 30 else")
parser.add_argument('--dim_med', type=int, default=53)
parser.add_argument('--dim_z', type=int, default=53)
parser.add_argument('--eta', type=float, default=0.1)
parser.add_argument('--zeta', type=float, default=0.1)
parser.add_argument('--step', type=int, default=2)
parser.add_argument('--mu', type=float, default=0.2)
parser.add_argument('--lambda_1', type=float, default=1)
parser.add_argument('--lambda_2', type=float, default=1)
parser.add_argument('--c_mlp', type=bool, default=True)
parser.add_argument('--epoch_threshold', type=int, default=50)

opt = parser.parse_args()

if opt.seed:
    print("Random Seed: ", opt.seed)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
torch.set_printoptions(threshold=float('inf'))

warnings.filterwarnings('ignore')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device:", device)
print("Dataset: ", opt.dataset)
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

'''load data'''
train_data = np.load("./datasets/{}/Clustering_s1234_graph70_train.npz".format(opt.dataset),
                     allow_pickle=True)
test_data = np.load("./datasets/{}/Clustering_s1234_graph70_test.npz".format(opt.dataset),
                    allow_pickle=True)

train_data, test_data = train_data["data"], test_data["data"]

print("data loaded.")

'''record loss result'''
log_dir = f"asset/log/{opt.dataset}.log"
writer = SummaryWriter(log_dir=log_dir)

'''initiate model'''
model = Geo(dim_in=opt.dim_in, dim_z=opt.dim_z, dim_med=opt.dim_med, dim_out=2, collaborative_mlp=opt.c_mlp)

print(opt)
model.apply(init_network_weights)
if cuda:
    model.cuda()


'''initiate criteria and optimizer'''
lr = opt.lr
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(opt.beta1, opt.beta2))
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

if __name__ == '__main__':
    train_data, test_data = get_data_generator(opt, train_data, test_data, normal=2)
    log_path = f"asset/log"
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    f = open(f"asset/log/{opt.dataset}.txt", 'a')
    f.write(f"\n*********{opt.dataset}*********\n")
    f.write("dim_in=" + str(opt.dim_in) + ", ")
    f.write("dim_z=" + str(opt.dim_z) + ", ")
    f.write("early_stop_epoch=" + str(opt.early_stop_epoch) + ", ")
    f.write("harved_epoch=" + str(opt.harved_epoch) + ", ")
    f.write("saved_epoch=" + str(opt.saved_epoch) + ", ")
    f.write("lr=" + str(opt.lr) + ", ")
    f.write("model_name=" + opt.model_name + ", ")
    f.write("seed=" + str(opt.seed) + ",")
    f.write("eta=" + str(opt.eta) + ", ")
    f.write("zeta=" + str(opt.zeta) + ", ")
    f.write("step=" + str(opt.step) + ", ")
    f.write("mu=" + str(opt.mu) + ", ")
    f.write("lambda1=" + str(opt.lambda_1) + ", ")
    f.write("lambda2=" + str(opt.lambda_2) + ", ")
    f.write("\n")
    f.close()

    # train
    losses = [np.inf]
    no_better_epoch = 0
    early_stop_epoch = 0

    epoch_threshold = opt.epoch_threshold

    for epoch in range(2000):
        print("epoch {}.    ".format(epoch))
        beta = min([(epoch * 1.) / max([100, 1.]), 1.])
        total_loss, total_mae, train_num, MSE_loss, XIB_KL_loss, KL_loss, AIB_KL_loss = 0, 0, 0, 0, 0, 0, 0
        model.train()
        count = 0
        for i in range(len(train_data)):
            count+=1

            lm_X, lm_Y, tg_X, tg_Y, y_max, y_min = train_data[i]["lm_X"], \
                                                                       train_data[i]["lm_Y"], \
                                                                       train_data[i]["tg_X"], \
                                                                       train_data[i]["tg_Y"], \
                                                                       train_data[i]["y_max"], \
                                                                       train_data[i]["y_min"]
            optimizer.zero_grad()
            #Geo
            # y_pred, _, _= model(Tensor(lm_X), Tensor(lm_Y), Tensor(tg_X),Tensor(tg_Y))

            #AIB
            y_pred, adj_mask_prob, adj_mask_matrix = model(Tensor(lm_X), Tensor(lm_Y), Tensor(tg_X),
                                                                      Tensor(tg_Y))

            distance = dis_loss(Tensor(tg_Y), y_pred, y_max, y_min)
            mse_loss = distance * distance  # mse loss
            mse_loss = mse_loss.sum()


            #Geo
            # loss = mse_loss

            #AIB
            AIB_prior = (0.5 * torch.ones_like(adj_mask_prob)).to(device)
            AIB_posterior = torch.distributions.bernoulli.Bernoulli(adj_mask_prob)
            AIB_prior = torch.distributions.bernoulli.Bernoulli(AIB_prior)
            aib_kl_loss = torch.distributions.kl.kl_divergence(AIB_posterior, AIB_prior)
            aib_kl_loss = opt.beta * aib_kl_loss.sum()
            loss = mse_loss + aib_kl_loss

            loss.backward()
            optimizer.step()

            MSE_loss += mse_loss.item()
            AIB_KL_loss +=  aib_kl_loss.item()
            total_loss += loss.item()
            total_mae += distance.sum()
            train_num += len(tg_Y)
          
            if epoch > 0 and epoch % 1 == 0:#print the first mask_matrix of epoch1
                    adj=[]
                    adj.append(adj_mask_matrix)
                    first_matrix = adj[0][-1]
                    print(first_matrix)
                    count_0 = adj.count(adj[0])
                    count_1 = adj.count(adj[1])
                    print(f"Count of elements at index 0: {count_0}")
                    print(f"Count of elements at index 1: {count_1}")

        AIB_KL_loss = AIB_KL_loss / train_num
        total_loss = total_loss / train_num
        MSE_loss = MSE_loss / train_num
        total_mae = total_mae / train_num

        writer.add_scalar('mse_loss/Train', MSE_loss, epoch)
        writer.add_scalar('AIB_KL_loss/Train', AIB_KL_loss, epoch)
        writer.add_scalar('total_loss/Train', total_loss, epoch)


        #
        print("train: total_loss: {:.4f} mae: {:.4f} mse_loss: {:.4f} AIB_kl_loss: {:.4f} ".format(total_loss,
                                                     total_mae, MSE_loss, AIB_KL_loss))

        # print("train: mse_loss: {:.4f} mae: {:.4f} ".format(MSE_loss, total_mae))#Geo

        # test
        total_mse, total_mae, test_num = 0, 0, 0
        dislist = []

        model.eval()
        distance_all = []
        with torch.no_grad():
            for i in range(len(test_data)):

                lm_X, lm_Y, tg_X, tg_Y, y_max, y_min = test_data[i]["lm_X"], \
                                                       test_data[i]["lm_Y"], \
                                                       test_data[i][ "tg_X"], \
                                                       test_data[i]["tg_Y"], \
                                                       test_data[i]["y_max"], \
                                                       test_data[i]["y_min"]

                y_pred,_,_ = model(Tensor(lm_X), Tensor(lm_Y), Tensor(tg_X),
                                                          Tensor(tg_Y))
                distance = dis_loss(Tensor(tg_Y), y_pred, y_max, y_min)
                for i in range(len(distance.cpu().detach().numpy())):
                    dislist.append(distance.cpu().detach().numpy()[i])
                    distance_all.append(distance.cpu().detach().numpy()[i])
                test_num += len(tg_Y)
                total_mse += (distance * distance).sum()
                total_mae += distance.sum()

            total_mse = total_mse / test_num
            total_mae = total_mae / test_num

            writer.add_scalar('MAE/Test', total_mae, epoch)

            print("test: mse: {:.4f}  mae: {:.4f}".format(total_mse, total_mae))
            dislist_sorted = sorted(dislist)
            print('test median:', dislist_sorted[int(len(dislist_sorted) / 2)])

            #save checkpoint for each 100 epochs
            if epoch > 0 and epoch % opt.saved_epoch == 0 and epoch < 1000:
                savepath = f"asset/model/{opt.dataset}_{epoch}.pth"
                save_cpt(model, optimizer, epoch, savepath)
                print("Save checkpoint!")


            batch_metric = total_mae.cpu().numpy()
            if batch_metric <= np.min(losses):
                no_better_epoch = 0
                early_stop_epoch = 0
                print("Better MAE in epoch {}: {:.4f}".format(epoch, batch_metric))
            else:
                no_better_epoch = no_better_epoch + 1
                early_stop_epoch = early_stop_epoch + 1

            losses.append(batch_metric)

            # halve the learning rate
        if no_better_epoch % opt.harved_epoch == 0 and no_better_epoch != 0:
            lr /= 2
            print("learning rate changes to {}!\n".format(lr))
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(opt.beta1, opt.beta2))
            no_better_epoch = 0

        if early_stop_epoch == opt.early_stop_epoch:
            savepath = f"asset/model/{opt.dataset}_{epoch}.pth"
            save_cpt(model, optimizer, epoch, savepath)
            print("Save checkpoint!")

            f = open(f"asset/log/{opt.dataset}.txt", 'a')
            f.write(f"\n*********epoch={epoch}*********\n")
            f.write("test: mse: {:.3f}\tmae: {:.3f}".format(total_mse, total_mae))
            f.write("\ttest median: {:.3f}".format(dislist_sorted[int(len(dislist_sorted) / 2)]))
            f.close()
            break
