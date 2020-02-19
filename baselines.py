from collections import OrderedDict 

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import dgl
import dgl.function as fn

from blocks import EdgeBlock, NodeBlock, GlobalBlock
from modules import GNLayer

from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor


# baselines
class LinearRegressionImputation(nn.Module):
    def __init__(self,
                 input_node_dim,
                 output_node_dim,
                 ):
        super(LinearRegressionImputation, self).__init__()
        
        self.input_node_dim = input_node_dim
        self.output_node_dim = output_node_dim
        self.fake_param = nn.Parameter(torch.zeros(1, 1)) # only for compatibility
        
    def forward(self, rgn_graph_input_batch_list, graph_input_batch):
        num_steps = len(rgn_graph_input_batch_list)

        all_input = torch.stack([x.ndata['x'] for x in rgn_graph_input_batch_list], dim=0).data.cpu().numpy() # T x N x F
        all_output_X = graph_input_batch.ndata['x'].data.cpu().numpy() # N x F
        node_num = all_input.shape[1]
        output_ys = []
        for node_i in range(node_num):
            Xy = all_input[:, node_i, :]
            X = np.concatenate([Xy[:, self.output_node_dim:], np.arange(all_input.shape[0], dtype=np.float32).reshape((-1, 1))], axis=-1)
            X = X[..., -1:]
            y = Xy[:, :self.output_node_dim]
            
            # remove missing input
            mask = Xy[..., -1]
            nomaskids = np.where(mask == 0)[0]
            if len(nomaskids > 1):
                X = X[nomaskids]
                y = y[nomaskids]
            else:
                print(nomaskids)

            reg = LinearRegression().fit(X, y)
            output_X = np.concatenate([all_output_X[node_i, self.output_node_dim:].reshape(1, -1), np.array(all_input.shape[0], dtype=np.float32).reshape((1, 1))], axis=-1)
            output_X = output_X[..., -1:]
            output_y = reg.predict(output_X) # 1 x F
            output_ys.append(output_y)
        output_ys = np.concatenate(output_ys, axis=0)
        output_ys = torch.from_numpy(output_ys).to(graph_input_batch.ndata['x'].device).float().contiguous()
        graph_input_batch.ndata['h_v'] = output_ys
        return graph_input_batch, None


class GRUImputation(nn.Module):
    def __init__(self, input_node_dim, output_node_dim, hidden_dim, layer_num):
        super(GRUImputation, self).__init__()
        self.input_node_dim = input_node_dim
        self.output_node_dim = output_node_dim
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.gru = nn.GRU(self.input_node_dim, self.hidden_dim, self.layer_num)
        self.out = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_node_dim)
        )

    def forward(self, rgn_graph_input_batch_list, graph_input_batch):
        num_steps = len(rgn_graph_input_batch_list)
        all_input = torch.stack([x.ndata['x'] for x in rgn_graph_input_batch_list], dim=0) # T x N x F
        init_h = all_input.new_zeros(self.layer_num, all_input.shape[1], self.hidden_dim)
        h = init_h
        _, h = self.gru(all_input, h)
        last_input = graph_input_batch.ndata['x'].unsqueeze(0)
        output, _ = self.gru(last_input, h)
        out_v = self.out(output[0])
        graph_input_batch.ndata['h_v'] = out_v
        return graph_input_batch, None


class KNNImputation(nn.Module):
    def __init__(self, input_node_dim, output_node_dim):
        super(KNNImputation, self).__init__()
        self.input_node_dim = input_node_dim
        self.output_node_dim = output_node_dim
        self.imputer = KNNImputer()
        self.fake_param = nn.Parameter(torch.zeros(1, 1)) # only for compatibility

    def forward(self, rgn_graph_input_batch_list, graph_input_batch):
        num_steps = len(rgn_graph_input_batch_list)

        all_input = torch.stack([x.ndata['x'] for x in rgn_graph_input_batch_list], dim=0).data.cpu().numpy() # T x N x F
        all_output_X = graph_input_batch.ndata['x'].data.cpu().numpy() # N x F
        all_input = np.concatenate((all_input, np.expand_dims(all_output_X, 0)), axis=0) # T x N x F
        # feature-wise imputation
        imputed_features = []
        for feat_i in range(self.output_node_dim):
            feature_X = all_input[..., feat_i].transpose((1, 0)) # N x T
            mask = all_input[..., -1].transpose((1, 0)) # N x T
            feature_X[np.where(mask == 1)] = np.nan
            feature_X = self.imputer.fit_transform(feature_X)
            imputed_features.append(feature_X.transpose((1, 0)))
        imputed_features = np.stack(imputed_features, axis=-1) # T x N x F
        imputed_features_last = imputed_features[-1]
        imputed_features_last = torch.from_numpy(imputed_features_last).to(graph_input_batch.ndata['x'].device).float().contiguous()
        graph_input_batch.ndata['h_v'] = imputed_features_last
        return graph_input_batch, None


class HHopImputation(nn.Module):
    '''Average over all h-hop neighbors for imputation
    '''
    def __init__(self, input_node_dim, output_node_dim, h_hop=1):
        super(HHopImputation, self).__init__()
        self.input_node_dim = input_node_dim
        self.output_node_dim = output_node_dim
        self.h_hop = h_hop
        self.fake_param = nn.Parameter(torch.zeros(1, 1)) # only for compatibility

    def forward(self, rgn_graph_input_batch_list, graph_input_batch):
        device = graph_input_batch.ndata['x'].device
        nodenum = graph_input_batch.number_of_nodes()
        imp_values = []
        for node_i in range(nodenum):
            neighbors = [torch.tensor([node_i,], device=device).long(),]
            for hopi in range(self.h_hop):
                last_neighbors = neighbors[-1]
                this_neighbors = torch.tensor([], device=device).long()
                for ni in last_neighbors:
                    this_neighbors = torch.cat((this_neighbors, graph_input_batch.predecessors(ni)))
                    this_neighbors = torch.cat((this_neighbors, graph_input_batch.successors(ni)))
                    this_neighbors = torch.unique(this_neighbors)
                    # this_neighbors_mask = graph_input_batch.ndata['x'][this_neighbors][:, -1]
                    # this_neighbors = this_neighbors[torch.where(this_neighbors_mask == 0)]
                neighbors.append(this_neighbors)
            neighbors = torch.unique(torch.cat(neighbors))
            neighbors_mask = graph_input_batch.ndata['x'][neighbors][:, -1]
            neighbors = neighbors[torch.where(neighbors_mask == 0)]
            imp_value = torch.mean(graph_input_batch.ndata['x'][neighbors][:, :self.output_node_dim], dim=0)
            imp_values.append(imp_value)
        imp_values = torch.stack(imp_values, dim=0)
        graph_input_batch.ndata['h_v'] = imp_values
        return graph_input_batch, None


class ExtraTreesImputation(nn.Module):
    def __init__(self, input_node_dim, output_node_dim):
        super(ExtraTreesImputation, self).__init__()
        self.input_node_dim = input_node_dim
        self.output_node_dim = output_node_dim
        self.imputer = IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=10, random_state=0), random_state=0)

        self.fake_param = nn.Parameter(torch.zeros(1, 1))

    def forward(self, rgn_graph_input_batch_list, graph_input_batch):
        num_steps = len(rgn_graph_input_batch_list)

        all_input = torch.stack([x.ndata['x'] for x in rgn_graph_input_batch_list], dim=0).data.cpu().numpy() # T x N x F
        all_output_X = graph_input_batch.ndata['x'].data.cpu().numpy() # N x F
        all_input = np.concatenate((all_input, np.expand_dims(all_output_X, 0)), axis=0) # T x N x F
        # feature-wise imputation
        imputed_features = []
        for feat_i in range(self.output_node_dim):
            feature_X = all_input[..., feat_i].transpose((1, 0)) # N x T
            mask = all_input[..., -1].transpose((1, 0)) # N x T
            feature_X[np.where(mask == 1)] = np.nan
            feature_X = self.imputer.fit_transform(feature_X)
            imputed_features.append(feature_X.transpose((1, 0)))
        imputed_features = np.stack(imputed_features, axis=-1) # T x N x F
        imputed_features_last = imputed_features[-1]
        imputed_features_last = torch.from_numpy(imputed_features_last).to(graph_input_batch.ndata['x'].device).float().contiguous()
        graph_input_batch.ndata['h_v'] = imputed_features_last
        return graph_input_batch, None