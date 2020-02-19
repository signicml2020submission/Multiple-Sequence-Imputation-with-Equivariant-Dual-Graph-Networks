from __future__ import division
from __future__ import print_function

import sys
import os
from itertools import chain
import multiprocessing as mpc

import logging
import pprint
import datetime
import yaml
import json

import numpy as np
import networkx as nx
import dgl

import torch
from torch.utils.data.dataset import Dataset

from sklearn.neighbors import kneighbors_graph


class NOAADataset(Dataset):
    def __init__(self, eventIds, graphs, X_seqs, edge_attrs, mask_seqs, cfg, trainset=False, train_additional_mask_rate=0.0):
        super(NOAADataset, self).__init__()
        assert all(len(x) == len(eventIds) for x in [eventIds, X_seqs, edge_attrs, mask_seqs])
        assert (len(graphs) == 0) or (len(graphs) == len(eventIds))
        self.eventIds = eventIds
        self.graphs = graphs
        self.X_seqs = X_seqs
        self.edge_attrs = edge_attrs
        self.mask_seqs = mask_seqs
        self.input_seq_length = cfg['train']['input_seq_length']
        self.filled_rgn = cfg['train']['filled_rgn']
        self.step_num = len(X_seqs[0])
        self.total = len(self.eventIds)
        self.cfg = cfg
        self.trainset = trainset
        self.train_additional_mask_rng = np.random.RandomState(cfg['seed'])
        self.train_additional_mask_rate = train_additional_mask_rate

    def __len__(self):
        return self.total

    def __getitem__(self, index):
        event_index = index
        eventId = self.eventIds[event_index]
        graph_seq = self.graphs[event_index]
        X_seq = self.X_seqs[event_index] # T x N x F
        edge_attr = self.edge_attrs[event_index]
        mask_seq = self.mask_seqs[event_index]
        # Not sequentially trained. Sample input window and target
        input_batch = []
        input_X_seq_batch = []
        mask_batch = []
        steps = X_seq.shape[0]
        input_seq_length = self.input_seq_length
        for jj in range(steps-input_seq_length+1):
            # sampling starting timestamp
            # if self.trainset:
            if False:
                st = np.random.randint(steps-input_seq_length+1)
            else:
                st = jj
            tt = st + input_seq_length - 1

            #### Data setting
            mask = mask_seq[tt]
            input_X_seq = X_seq[st:tt+1]
            input_X_seq_batch.append(input_X_seq)
            graph_input = graph_seq[st:tt+1]    # input graph sequence (window)
            for ll in range(input_seq_length):
                _graph = graph_input[ll]
                _X = input_X_seq[ll]
            try:
                for hh in range(num_spatial_hops):
                    graph_input[0].ndata.pop('h'+str(hh)+'_v')
                    graph_input[0].edata.pop('h'+str(hh)+'_e')
            except:
                pass

            #### Optimize on one segment
            rgn_graph_input = graph_input[:-1]    # 0:t-1
            sgn_graph_input = graph_input[-1]     # t

            if self.trainset and self.train_additional_mask_rate > 0:
                ### additional masked nodes
                nodenum = sgn_graph_input.number_of_nodes()
                not_masked_nodes = []
                for ni in range(nodenum-1):
                    if ni not in mask:
                        not_masked_nodes.append(ni)
                not_masked_nodes = np.array(not_masked_nodes, dtype=np.int64)
                train_additional_node_num = int(nodenum * self.train_additional_mask_rate)
                train_additional_mask = self.train_additional_mask_rng.choice(not_masked_nodes, train_additional_node_num, replace=False)
                train_additional_mask = torch.from_numpy(train_additional_mask).long().contiguous()
                sample_mask_x = sgn_graph_input.ndata['x'][mask[0]]
                mask = torch.cat((mask, train_additional_mask))
                temp_tensor = sgn_graph_input.ndata['x'][mask]
                temp_tensor[:, 0] = sample_mask_x[0]
                temp_tensor[:, -2:] = torch.tensor([0., 1.]).float()
                sgn_graph_input.ndata['x'][mask] = temp_tensor

            mask_batch.append(mask)
            input_batch.append((rgn_graph_input, sgn_graph_input))

        rgn_graph_input_list = [x[0] for x in input_batch]
        rgn_graph_input_batch_list = [[x[jj] for x in rgn_graph_input_list] for jj in range(input_seq_length - 1)]
        sgn_graph_input_batch = [x[1] for x in input_batch]
        input_X_seq_batch = torch.stack([x[-1] for x in input_X_seq_batch], dim=0)
        mask_batch = torch.stack(mask_batch, dim=0)

        return rgn_graph_input_batch_list, sgn_graph_input_batch, input_X_seq_batch, mask_batch


def noaa_collate_fn(batch):
    rgn_graph_input_batch_lists, sgn_graph_input_batchs, input_X_seq_batchs, mask_batchs = list(zip(*batch))
    rgn_graph_input_batch_list = [dgl.batch(list(chain.from_iterable(s))) for s in zip(*rgn_graph_input_batch_lists)]
    sgn_graph_input_batch = dgl.batch(list(chain.from_iterable(sgn_graph_input_batchs)))
    input_X_seq_batch = torch.cat(input_X_seq_batchs, dim=0)
    mask_batch = torch.cat(mask_batchs, dim=0)
    return rgn_graph_input_batch_list, sgn_graph_input_batch, input_X_seq_batch, mask_batch


def load_noaa_data(cfg):
    # hard-code pre-defined node sets
    TEST_NODE_SETS = {
        'interpolation_10': [
            24, 94, 11, 45, 43, 51, 61,
            2, 6, 64, 121, 114, 168, 76,
            86, 148, 134, 176, 188
        ],
        'extrapolation_10': [
            155, 156, 157, 158, 159, 172, 176,
            177, 178, 179, 180, 181, 182, 183,
            184, 185, 186, 187, 188, 189, 190
        ],
        'interpolation_25': [
            24, 94, 11, 45, 43, 51, 61,
            2, 6, 64, 121, 114, 168, 76,
            86, 148, 134, 176, 188,
            84, 85, 90, 89, 130, 131, 144, 147,
            34,36,37,15,17,57,58,18,0,12,104,
            164,140,138,142,141,143,171,56,97
        ],
        'extrapolation_25': [134,138,139,140,141,142,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190]
    }

    dataset_path = cfg['dataset']['dataset_path']
    input_seq_length = cfg['train']['input_seq_length']

    node_meta = np.load(os.path.join(dataset_path, 'node_meta.npy'), allow_pickle=True)
    node_meta = torch.from_numpy(node_meta).float()
    edge_index = np.load(os.path.join(dataset_path, 'edge_index_4nn.npy'), allow_pickle=True)
    seqvals = []
    for fn in sorted(os.listdir(dataset_path)):
        if fn.endswith('_seq.npz'):
            seqdata = np.load(os.path.join(dataset_path, fn))
            seqvals.append(seqdata['vals'][:, :, 1:2]) # load temperature only

    eventIds = torch.arange(len(seqvals)).long()
    ref_graph = dgl.DGLGraph()
    num_nodes = node_meta.shape[0]
    cfg['dataset']['num_nodes'] = num_nodes
    cfg['dataset']['num_edge_types'] = 1
    ref_graph.add_nodes(num_nodes)
    ref_graph.add_edges(edge_index[0], edge_index[1])

    missing_rate = cfg['dataset']['missing_rate']
    num_missing = int(missing_rate*num_nodes)
    num_checked_seqs = cfg['dataset']['num_checked_seqs']

    if ('node_unseen_rate' in cfg['dataset']) or ('node_unseen_set' in cfg['dataset']):
        # remove some nodes from training and validation
        if ('node_unseen_rate' in cfg['dataset']):
            node_unseen_rate = cfg['dataset']['node_unseen_rate']
            node_unseen_num = int(node_unseen_rate * num_nodes)
            shuffled_node_ids = np.random.permutation(np.arange(num_nodes, dtype=np.int64))
            node_unseen_ids, node_seen_ids = shuffled_node_ids[:node_unseen_num], shuffled_node_ids[node_unseen_num:]
        elif ('node_unseen_set' in cfg['dataset']):
            node_unseen_ids = TEST_NODE_SETS[cfg['dataset']['node_unseen_set']]
            node_seen_ids = [x for x in range(num_nodes) if x not in node_unseen_ids]
            node_unseen_ids, node_seen_ids = np.array(node_unseen_ids, dtype=np.int64), np.array(node_seen_ids, dtype=np.int64)

        node_meta_utm = np.load(os.path.join(dataset_path, 'node_meta_utm.npy'), allow_pickle=True)
        new_node_meta_utm = node_meta_utm[node_seen_ids]
        new_knn_graph_array = kneighbors_graph(new_node_meta_utm, n_neighbors=4).toarray()
        new_knn_graph_array = new_knn_graph_array + np.transpose(new_knn_graph_array)
        new_knn_graph_edge_index = np.array(np.where(new_knn_graph_array > 0))

        if 'limit_node_num_added' in cfg['dataset']:
            limit_node_num_added = cfg['dataset']['limit_node_num_added']
            node_unseen_ids = node_unseen_ids[:limit_node_num_added]
            all_node_ids = np.concatenate((node_seen_ids, node_unseen_ids))
            all_node_meta_utm = node_meta_utm[all_node_ids]
            all_knn_graph_array = kneighbors_graph(all_node_meta_utm, n_neighbors=4).toarray()
            all_knn_graph_array = all_knn_graph_array + np.transpose(all_knn_graph_array)
            all_knn_graph_edge_index = np.array(np.where(all_knn_graph_array > 0))
        else:
            all_node_ids = None
    else:
        node_seen_ids = None

    eventIds = []
    graph_seqs = []
    X_seqs = []
    edge_attrs = []
    mask_seqs = []

    num_seqs = 0
    for seqval in seqvals:
        num_seqs += (seqval.shape[0] - input_seq_length + 1)
    tr_ind = int(cfg['dataset']['tr_ratio']*num_seqs)
    val_ind = tr_ind + int(cfg['dataset']['val_ratio']*num_seqs)

    tmp_seq_nums = []
    for seqval in seqvals:
        if (node_seen_ids is not None) and (len(graph_seqs) < val_ind):
            graph_seqs_list, X_seqs_list, edge_attrs_list, mask_seqs_list = load_processed_seq(
                seqval[:, node_seen_ids, :], node_meta[node_seen_ids], new_knn_graph_edge_index, input_seq_length, missing_rate)
        elif (node_seen_ids is not None) and (all_node_ids is not None) and (len(graph_seqs) >= val_ind):
            tmp_seq_nums.append(len(graph_seqs))
            graph_seqs_list, X_seqs_list, edge_attrs_list, mask_seqs_list = load_processed_seq(
                seqval[:, all_node_ids, :], node_meta[all_node_ids], all_knn_graph_edge_index, input_seq_length, missing_rate)
        else:
            tmp_seq_nums.append(len(graph_seqs))
            graph_seqs_list, X_seqs_list, edge_attrs_list, mask_seqs_list = load_processed_seq(
                seqval, node_meta, edge_index, input_seq_length, missing_rate)
        graph_seqs.extend(graph_seqs_list)
        X_seqs.extend(X_seqs_list)
        edge_attrs.extend(edge_attrs_list)
        mask_seqs.extend(mask_seqs_list)
    eventIds = list(range(len(graph_seqs)))
    if node_seen_ids is not None:
        val_ind = tmp_seq_nums[0]
        for gid in range(val_ind, len(graph_seqs)):
            for idx_graph in range(len(graph_seqs[gid])):
                # completely missing features and coordinates
                # graph_seqs[gid][idx_graph].ndata['x'][node_unseen_ids] = torch.tensor([0., 0., 0., 0., 0., 1.]).float()

                # completely missing features but not coordinates
                # temp_tensor = graph_seqs[gid][idx_graph].ndata['x'][node_unseen_ids]
                # temp_tensor[:, 0] = 0
                # temp_tensor[:, -2:] = torch.tensor([0., 1.]).float()
                # graph_seqs[gid][idx_graph].ndata['x'][node_unseen_ids] = temp_tensor

                # randomly missing features but not coordinates (do nothing, new nodes miss like old nodes)
                pass

    X_seqs_train = torch.cat(X_seqs[:tr_ind], dim=0)
    X_seqs_train = X_seqs_train.reshape(-1, X_seqs_train.shape[-1])
    X_seqs_train_mean, X_seqs_train_std = torch.mean(X_seqs_train, dim=0), torch.std(X_seqs_train, dim=0)
    node_meta_mean, node_meta_std = torch.mean(node_meta, dim=0), torch.std(node_meta, dim=0)
    for idx in range(len(graph_seqs)):
        X_seqs[idx] = (X_seqs[idx] - X_seqs_train_mean) / X_seqs_train_std
        for idx_graph in range(len(graph_seqs[idx])):
            X_seq_dim = X_seqs_train_mean.shape[0]
            node_meta_dim = node_meta_mean.shape[0]
            graph_seqs[idx][idx_graph].ndata['x'][..., :X_seq_dim] -= X_seqs_train_mean
            graph_seqs[idx][idx_graph].ndata['x'][..., :X_seq_dim] /= X_seqs_train_std
            graph_seqs[idx][idx_graph].ndata['x'][..., X_seq_dim:X_seq_dim+node_meta_dim] -= node_meta_mean
            graph_seqs[idx][idx_graph].ndata['x'][..., X_seq_dim:X_seq_dim+node_meta_dim] /= node_meta_std
            # masked = torch.where(graph_seqs[idx][idx_graph].ndata['x'][..., -1] == 1)
            # graph_seqs[idx][idx_graph].ndata['x'][masked] = torch.tensor([0, 0, 0, 0, 0, 1]).float()

    if 'train_additional_mask_rate' in cfg['dataset']:
        train_additional_mask_rate = cfg['dataset']['train_additional_mask_rate']
    else:
        train_additional_mask_rate = 0.0
    train_dataset = NOAADataset(eventIds[:tr_ind], graph_seqs[:tr_ind],
        X_seqs[:tr_ind], edge_attrs[:tr_ind], mask_seqs[:tr_ind], cfg, trainset=True, train_additional_mask_rate=train_additional_mask_rate)
    valid_dataset = NOAADataset(eventIds[tr_ind:val_ind], graph_seqs[tr_ind:val_ind],
        X_seqs[tr_ind:val_ind], edge_attrs[tr_ind:val_ind], mask_seqs[tr_ind:val_ind], cfg)
    test_dataset = NOAADataset(eventIds[val_ind:], graph_seqs[val_ind:],
        X_seqs[val_ind:], edge_attrs[val_ind:], mask_seqs[val_ind:], cfg)

    print('test: {} nodes'.format(test_dataset[0][1][0].number_of_nodes()))

    retdict = {
        'dataset': {
            'train': train_dataset, 'valid': valid_dataset, 'test': test_dataset
        },
        'meta': {
            'num_nodes': num_nodes, 'num_edge_types': 1, 'missing_rate': missing_rate,
            'num_missing': num_missing, 'num_checked_seqs': num_checked_seqs
        }
    }
    if node_seen_ids is not None:
        retdict['node_unseen_ids'] = node_unseen_ids
    return retdict


def load_processed_seq(event_seqs, node_meta, edge_index, input_seq_length, missing_rate=0.5):

    num_nodes = node_meta.shape[0]
    ref_graph = dgl.DGLGraph()
    ref_graph.add_nodes(num_nodes)
    ref_graph.add_edges(edge_index[0], edge_index[1])
    num_missing = int(missing_rate * num_nodes)
    mask_seq = []
    X_seq = event_seqs.copy()    # Fully filled observations (# T x #nodes x #features)
    X_seq = torch.tensor(X_seq, dtype=torch.float32)
    for _ in range(X_seq.shape[0]):
        # Randomly choose nodes to make them missing
        sample_ind = np.random.choice(num_nodes-1, num_missing, replace=False)    # ball is not masked.
        mask_seq.append(sample_ind)

    # mask_seq = np.array(mask_seq)
    mask_seq = torch.tensor(mask_seq, dtype=torch.int64)

    #### Set missing masks
    #### Build graph sequence with the missing mask
    graph_seqs = []
    X_seqs = []
    edge_attrs = []
    mask_seqs = []
    for jj in range(X_seq.shape[0]-input_seq_length+1):
        st = jj
        tt = jj + input_seq_length - 1
        mask_seq_temp = mask_seq[st:tt+1]
        X_temp = X_seq[st:tt+1]
        graph_seqs_jj = []
        X_seqs_jj = []
        edge_attrs_jj = []
        mask_seqs_jj = []
        for mask, X in zip(mask_seq_temp, X_temp):
            tmp_graph = dgl.DGLGraph(ref_graph)    # fully connected graph
            edge_attr = torch.zeros(tmp_graph.number_of_edges()).long()
            tmp_graph.edata['etype'] = edge_attr    # For nn.Embedding
            X_withmeta = torch.cat((X, node_meta), dim=-1)
            tmp_graph.ndata['x'] = torch.cat([X_withmeta, torch.ones(X_withmeta.shape[0], 1), torch.zeros(X_withmeta.shape[0], 1)], dim=-1)    # #nodes x (#features + 3)
            # Setting one-hot encoding. [non_masked, masked]
            temp_tensor = tmp_graph.ndata['x'][mask]
            temp_tensor[:, 0] = 0
            temp_tensor[:, -2:] = torch.tensor([0., 1.]).float()
            tmp_graph.ndata['x'][mask] = temp_tensor
            # tmp_graph.ndata['x'][mask] = torch.tensor([0., 0., 0., 0., 0., 1.]).float()
            graph_seqs_jj.append(tmp_graph)
            X_seqs_jj.append(X)
            edge_attrs_jj.append(edge_attr)
            mask_seqs_jj.append(mask)
        graph_seqs.append(graph_seqs_jj)
        X_seqs.append(torch.stack(X_seqs_jj, dim=0))
        edge_attrs.append(torch.stack(edge_attrs_jj, dim=0))
        mask_seqs.append(torch.stack(mask_seqs_jj, dim=0))
    return graph_seqs, X_seqs, edge_attrs, mask_seqs
