from layers import *
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Geo(nn.Module):
    def __init__(self, dim_in, dim_z, dim_med, dim_out, collaborative_mlp=False):
        super(Geo, self).__init__()
        self.dim_z = dim_z

        self.att_attribute = SimpleAttention1(temperature=dim_z ** 0.5,
                                             d_q_in=dim_in,
                                             d_k_in=dim_in,
                                             d_v_in=dim_in + 2,
                                             d_q_out=dim_z,
                                             d_k_out=dim_z,
                                             d_v_out=dim_z)


        if collaborative_mlp:
            self.pred = SimpleAttention2(temperature=dim_z ** 0.5,
                                        d_q_in=dim_in * 2 + 2,
                                        d_k_in=dim_in,
                                        d_v_in=2,
                                        d_q_out=dim_z,
                                        d_k_out=dim_z,
                                        d_v_out=2,
                                        drop_last_layer=False)

        else:
            self.pred = nn.Sequential(
                nn.Linear(dim_z, dim_med),
                nn.ReLU(),
                nn.Linear(dim_med, dim_out)
            )

        self.collaborative_mlp = collaborative_mlp
        self.gat_layer = GATConv(
            in_channels=dim_in+2,  # Input feature dimension
            out_channels=32,  # Output feature dimension (you can adjust this)
            heads=1,  # Number of attention heads (you can adjust this)
            concat=False,  # Use 'True' for multi-head attention, 'False' for single head
            dropout=0.5,  # Dropout rate for attention weights (you can adjust this)
        )

        # calculate A
        self.gamma_1 = nn.Parameter(torch.ones(1, 1))
        self.gamma_2 = nn.Parameter(torch.ones(1, 1))
        self.gamma_3 = nn.Parameter(torch.ones(1, 1))
        self.alpha = nn.Parameter(torch.ones(1, 1))
        self.beta = nn.Parameter(torch.zeros(1, 1))

        # transform in Graph
        self.w_1 = nn.Linear(dim_in + 2, dim_in + 2)
        self.w_2 = nn.Linear(dim_in + 2, dim_in + 2)
        self.feat_mask = torch.rand(self.dim_z, requires_grad=True).to(device)

    def update_tg_feature(self, tg_feature, adjacency_matrix):

        graph_data = Data(
            x=tg_feature,  # Node feature matrix
            edge_index=torch.nonzero(adjacency_matrix).t().contiguous(),  # Edge indices
        )

        # Apply the GAT layer to propagate information and update tg_feature
        tg_feature = self.gat_layer(graph_data.x, graph_data.edge_index)

        return tg_feature

    def adj_mask(self, lm_X, tg_X, all_feature_0):

        combined_features = torch.cat((lm_X, tg_X), dim=0)
        _, mask_a_score = self.att_attribute(combined_features, combined_features, all_feature_0)
        mask_a_prob = torch.clamp(torch.sigmoid(mask_a_score), 0.001, 0.999)
        mask_a_matrix = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
            torch.Tensor([0.05]).cuda(),
            probs=mask_a_prob).rsample()
        eps = 0.5
        mask_a_matrix_1 = (mask_a_matrix > eps).detach().float()
        return mask_a_prob, mask_a_matrix

    def forward(self, lm_X, lm_Y, tg_X, tg_Y, reserve_ratio):

        N_X = len(lm_X)
        idx = torch.randperm(N_X)
        num_to_reserve = max(1, int(N_X * reserve_ratio))
        reserved_idx = idx[:num_to_reserve]
        reserved_idx_list = reserved_idx.tolist()
        lm_X = lm_X[reserved_idx_list, :]
        lm_Y = lm_Y[reserved_idx_list, :]


        N1 = lm_Y.size(0)
        N2 = tg_Y.size(0)
        ones = torch.ones(N1 + N2).cuda()
        lm_feature = torch.cat((lm_X, lm_Y), dim=1).cuda()
        tg_feature_0 = torch.cat((tg_X, torch.zeros(N2, 2).cuda()), dim=1)
        all_feature_0 = torch.cat((lm_feature, tg_feature_0), dim=0)

        adj_matrix_0 = torch.diag(ones)

        _, lm_attribute_score = self.att_attribute(lm_X, lm_X, lm_feature)
        lm_attribute_score = torch.exp(lm_attribute_score)
        _, attribute_score = self.att_attribute(tg_X, lm_X, lm_feature)
        attribute_score = torch.exp(attribute_score)

        # Geo
        # adj_matrix_0[N1:N1 + N2, :N1] = attribute_score
        # adj_matrix_0[:N1, :N1] = lm_attribute_score


        # AIB
        mask_a_prob, mask_a_matrix = self.adj_mask(lm_X, tg_X, all_feature_0)
        adj_matrix_0[N1:N1 + N2, :N1] = attribute_score * mask_a_matrix[N1:N1 + N2, :N1]


        degree_0 = torch.sum(adj_matrix_0, dim=1)
        # degree_0 = torch.add(degree_0, 1e-5)
        degree_reverse_0 = 1.0 / degree_0
        degree_matrix_reverse_0 = torch.diag(degree_reverse_0)

        degree_mul_adj_0 = degree_matrix_reverse_0 @ adj_matrix_0
        step_1_all_feature = self.w_1(degree_mul_adj_0 @ all_feature_0)
        tg_feature_1 = step_1_all_feature[N1:N1 + N2, :]

        final_tg_feature = torch.cat((tg_X,
                                      tg_feature_1), dim=-1)
        '''
        predict
        both normal mlp and collaborative mlp are ok, we suggest:
            (1) the number of neighbors > 10: using collaborative mlp
            (2) the number of neighbors < 10: using normal mlp
        '''

        if self.collaborative_mlp:
            y_pred, _ = self.pred(final_tg_feature, lm_X, lm_Y)

        else:
            y_pred = self.pred(final_tg_feature)

        # return y_pred, _, _#Geo
        return y_pred, mask_a_prob, mask_a_matrix#AIB
