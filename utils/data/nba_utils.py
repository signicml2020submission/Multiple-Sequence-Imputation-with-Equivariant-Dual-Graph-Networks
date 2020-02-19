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

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc


class NBADataset(Dataset):
    def __init__(self, eventIds, graphs, X_seqs, edge_attrs, mask_seqs, cfg, trainset=False):
        super(NBADataset, self).__init__()
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
            mask_batch.append(mask) 
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

            input_batch.append((rgn_graph_input, sgn_graph_input))
        
        rgn_graph_input_list = [x[0] for x in input_batch]
        rgn_graph_input_batch_list = [[x[jj] for x in rgn_graph_input_list] for jj in range(input_seq_length - 1)]
        sgn_graph_input_batch = [x[1] for x in input_batch]
        input_X_seq_batch = torch.stack([x[-1] for x in input_X_seq_batch], dim=0)
        mask_batch = torch.stack(mask_batch, dim=0)

        return rgn_graph_input_batch_list, sgn_graph_input_batch, input_X_seq_batch, mask_batch
        

def nba_collate_fn(batch):
    rgn_graph_input_batch_lists, sgn_graph_input_batchs, input_X_seq_batchs, mask_batchs = list(zip(*batch))
    rgn_graph_input_batch_list = [dgl.batch(list(chain.from_iterable(s))) for s in zip(*rgn_graph_input_batch_lists)]
    sgn_graph_input_batch = dgl.batch(list(chain.from_iterable(sgn_graph_input_batchs)))
    input_X_seq_batch = torch.cat(input_X_seq_batchs, dim=0)
    mask_batch = torch.cat(mask_batchs, dim=0)
    return rgn_graph_input_batch_list, sgn_graph_input_batch, input_X_seq_batch, mask_batch


def load_nba_data(cfg):
    dataset = json.load(open(cfg['dataset']['dataset_path'], "r"))
    logging.info("Dataset: {} is loaded.".format(cfg['dataset']['dataset_path']))

    playerid_dict = json.load(open(cfg['dataset']['playerid_path'],'r'))
    tmp = {}
    {tmp.update({int(k): v}) for k,v in playerid_dict.items()}
    playerid_dict.update(tmp)
    logging.info("Active playerid_dict: {} is loaded.".format(cfg['dataset']['dataset_path']))

    num_nodes = cfg['dataset']['num_nodes']
    num_edge_types = cfg['dataset']['num_edge_types']
    missing_rate = cfg['dataset']['missing_rate']
    num_missing = int(missing_rate*num_nodes)
    num_checked_seqs = cfg['dataset']['num_checked_seqs']

    # p = mpc.Pool(8)
    results = []
    for gameId in list(dataset.keys()):
        # results.append(p.apply_async(load_processed_game, args=(dataset, playerid_dict, gameId, missing_rate, num_checked_seqs)))
        results.append(load_processed_game(dataset, playerid_dict, gameId, missing_rate, num_checked_seqs))
    # p.close()
    # p.join()
    # results = [x.get() for x in results]
    eventIds, graph_seqs, X_seqs, edge_attrs, mask_seqs = [list(chain.from_iterable(x)) for x in list(zip(*results))]

    logging.info("{} sequences are processed and loaded".format(len(graph_seqs)))
    logging.info("# nodes: {}".format(num_nodes))
    ###################################################
    
    
    ########## Generate graph sequence and missing values ##########
    num_seqs = len(graph_seqs)
    tr_ind = int(cfg['dataset']['tr_ratio']*num_seqs)
    val_ind = tr_ind + int(cfg['dataset']['val_ratio']*num_seqs)


    logging.info("[Training] Graph sequence length: {}".format(tr_ind))
    logging.info("[Validation] Graph sequence length: {}".format(val_ind - tr_ind))
    logging.info("[Test] Graph sequence length: {}".format(num_seqs - val_ind))
    ###################################################

    train_dataset = NBADataset(eventIds[:tr_ind], graph_seqs[:tr_ind], 
        X_seqs[:tr_ind], edge_attrs[:tr_ind], mask_seqs[:tr_ind], cfg, trainset=True)
    valid_dataset = NBADataset(eventIds[tr_ind:val_ind], graph_seqs[tr_ind:val_ind], 
        X_seqs[tr_ind:val_ind], edge_attrs[tr_ind:val_ind], mask_seqs[tr_ind:val_ind], cfg)
    test_dataset = NBADataset(eventIds[val_ind:], graph_seqs[val_ind:], 
        X_seqs[val_ind:], edge_attrs[val_ind:], mask_seqs[val_ind:], cfg)

    return {
        'dataset': {
            'train': train_dataset, 'valid': valid_dataset, 'test': test_dataset
        },
        'meta': {
            'num_nodes': num_nodes, 'num_edge_types': num_edge_types, 'missing_rate': missing_rate,
            'num_missing': num_missing, 'num_checked_seqs': num_checked_seqs
        }
    }


def load_processed_game(dataset, playerid_dict, gameId, missing_rate, num_checked_seqs):
    eventIds, graph_seqs, X_seqs, edge_attrs, mask_seqs = [], [], [], [], []
    for eventId in list(dataset[gameId].keys())[:num_checked_seqs]:
        try:
            graph_seq, X_seq, edge_attr, mask_seq = load_processed_seq(dataset, playerid_dict, gameId, eventId, missing_rate)
            logging.info("[Loaded sequence] Graph sequence length of gameId ({}), eventId ({}): {}".format(gameId, eventId, len(graph_seq)))
            eventIds.append(eventId)
            graph_seqs.append(graph_seq)
            X_seqs.append(X_seq)
            edge_attrs.append(edge_attr)
            mask_seqs.append(mask_seq)
        except:
            continue
    return eventIds, graph_seqs, X_seqs, edge_attrs, mask_seqs


def get_player_name(playerid_dict, playerId):
    if isinstance(playerId, str):
        return playerid_dict[playerId]
    elif isinstance(playerId, int):
        return playerid_dict[str(playerId)]
    

def get_player_seq_per_game_event(dataset, gameId, eventId, playerId, ndarray=True):
    if isinstance(eventId, int):
        eventId = str(eventId)
    if isinstance(playerId, int):
        playerId = str(playerId)
        
    if ndarray:
        return np.asarray(dataset[gameId][eventId][playerId])
    else:
        return dataset[gamdId][eventId][playerId]


def get_event_seq_per_game(dataset, gameId, eventId, ndarray=True, normalize=False):
    if isinstance(eventId, int):
        eventId = str(eventId)
    pids, tmp = [], []
    if not dataset[gameId][eventId].keys():
        raise ValueError("gameId {}, eventId {} don't have any players.".format(gameId, eventId))

    for pid in sorted(dataset[gameId][eventId].keys(), reverse=True):    # ball(-1) will be the last element

        tmp.append(dataset[gameId][eventId][pid])
        pids.append(pid)
    
    tmp = np.array(tmp)
    if normalize:
        tmp[:,:,0] = tmp[:,:,0]/94
        tmp[:,:,1] = tmp[:,:,1]/50
    return pids, np.swapaxes(tmp, 0, 1)    # return T x N x D


def get_edge_attr(pids, playerid_dict, graph):
    edge_attr = []
#     type_edge = dict()
    type_edge = {'same_team': 0, 'diff_team': 1, 'to_ball': 2, 'from_ball': 3}
    
    for sid, rid in zip(graph.edges()[0], graph.edges()[1]):
        sid, rid = sid.item(), rid.item()
        spid, rpid = pids[sid], pids[rid]
        steam, rteam = playerid_dict[spid][1], playerid_dict[rpid][1]
        
        if steam == "ball":
            key = "from_ball"
        elif rteam == "ball":
            key = "to_ball"
        else:
            if steam == rteam:
                key = "same_team"
            else:
                key = "diff_team"
        
#         pair = (steam, rteam)
#         if not pair in type_edge:
#             type_edge[pair] = len(type_edge)

        edge_attr.append(type_edge[key])
    
    return np.array(edge_attr)


def load_processed_seq(dataset, playerid_dict, gameId, eventId, missing_rate=0.5):

    num_nodes = 10+1    # it is fixed. # players + 1 ball
    nx_g = nx.complete_graph(num_nodes)
    ref_graph = dgl.DGLGraph(nx_g)

    #### Get a sequence (per gameId, eventId)
    pids, event_seqs = get_event_seq_per_game(dataset, gameId, eventId, normalize=True)
    edge_attr = get_edge_attr(pids, playerid_dict, ref_graph)
    num_edge_types = len(set(edge_attr))

    #### Sampling missing nodes for each timestamp
    num_missing = int(missing_rate*num_nodes)
    mask_seq = []
    X_seq = event_seqs[:,:,:2].copy()    # Fully filled observations (# T x #nodes x #features)
    X_seq = torch.tensor(X_seq, dtype=torch.float32)
    for _ in range(X_seq.shape[0]):
        # Randomly choose nodes to make them missing
        sample_ind = np.random.choice(num_nodes-1, num_missing, replace=False)    # ball is not masked.
        mask_seq.append(sample_ind)

    # mask_seq = np.array(mask_seq)
    mask_seq = torch.tensor(mask_seq, dtype=torch.int64)

    #### Set missing masks
    #### Build graph sequence with the missing mask
    graph_seq = []
    for mask, X in zip(mask_seq, X_seq):
        
        tmp_graph = dgl.DGLGraph(nx_g)    # fully connected graph
    #     tmp_graph.edata['e'] = torch.tensor(np.expand_dims(edge_attr, axis=1), dtype=torch.float32, device=device)
        tmp_graph.edata['etype'] = torch.tensor(edge_attr, dtype=torch.long)    # For nn.Embedding
        tmp_graph.ndata['x'] = torch.cat([X[:,:2], torch.ones(X.shape[0], 1), torch.zeros(X.shape[0], 2)], axis=-1)    # #nodes x (#features + 3)

        # Setting one-hot encoding. [non_masked player, ball, masked_player]
        tmp_graph.ndata['x'][-1, 2:] = torch.tensor([0., 1., 0.])    # 
        tmp_graph.ndata['x'][mask] = torch.tensor([0., 0., 0., 0., 1.])

        graph_seq.append(tmp_graph)

    return graph_seq, X_seq, edge_attr, mask_seq


# Function to draw the basketball court lines
def draw_court(ax=None, color="gray", lw=1, zorder=0):
    """
    Draw court
    """
    
    if ax is None:
        ax = plt.gca()

    # Creates the out of bounds lines around the court
    outer = Rectangle((0,-50), width=94, height=50, color=color,
                      zorder=zorder, fill=False, lw=lw)

    # The left and right basketball hoops
    l_hoop = Circle((5.35,-25), radius=.75, lw=lw, fill=False, 
                    color=color, zorder=zorder)
    r_hoop = Circle((88.65,-25), radius=.75, lw=lw, fill=False,
                    color=color, zorder=zorder)
    
    # Left and right backboards
    l_backboard = Rectangle((4,-28), 0, 6, lw=lw, color=color,
                            zorder=zorder)
    r_backboard = Rectangle((90, -28), 0, 6, lw=lw,color=color,
                            zorder=zorder)

    # Left and right paint areas
    l_outer_box = Rectangle((0, -33), 19, 16, lw=lw, fill=False,
                            color=color, zorder=zorder)    
    l_inner_box = Rectangle((0, -31), 19, 12, lw=lw, fill=False,
                            color=color, zorder=zorder)
    r_outer_box = Rectangle((75, -33), 19, 16, lw=lw, fill=False,
                            color=color, zorder=zorder)

    r_inner_box = Rectangle((75, -31), 19, 12, lw=lw, fill=False,
                            color=color, zorder=zorder)

    # Left and right free throw circles
    l_free_throw = Circle((19,-25), radius=6, lw=lw, fill=False,
                          color=color, zorder=zorder)
    r_free_throw = Circle((75, -25), radius=6, lw=lw, fill=False,
                          color=color, zorder=zorder)

    # Left and right corner 3-PT lines
    # a represents the top lines
    # b represents the bottom lines
    l_corner_a = Rectangle((0,-3), 14, 0, lw=lw, color=color,
                           zorder=zorder)
    l_corner_b = Rectangle((0,-47), 14, 0, lw=lw, color=color,
                           zorder=zorder)
    r_corner_a = Rectangle((80, -3), 14, 0, lw=lw, color=color,
                           zorder=zorder)
    r_corner_b = Rectangle((80, -47), 14, 0, lw=lw, color=color,
                           zorder=zorder)
    
    # Left and right 3-PT line arcs
    l_arc = Arc((5,-25), 47.5, 47.5, theta1=292, theta2=68, lw=lw,
                color=color, zorder=zorder)
    r_arc = Arc((89, -25), 47.5, 47.5, theta1=112, theta2=248, lw=lw,
                color=color, zorder=zorder)

    # half_court
    # ax.axvline(470)
    half_court = Rectangle((47,-50), 0, 50, lw=lw, color=color,
                           zorder=zorder)

    hc_big_circle = Circle((47, -25), radius=6, lw=lw, fill=False,
                           color=color, zorder=zorder)
    hc_sm_circle = Circle((47, -25), radius=2, lw=lw, fill=False,
                          color=color, zorder=zorder)

    court_elements = [l_hoop, l_backboard, l_outer_box, outer,
                      l_inner_box, l_free_throw, l_corner_a,
                      l_corner_b, l_arc, r_hoop, r_backboard, 
                      r_outer_box, r_inner_box, r_free_throw,
                      r_corner_a, r_corner_b, r_arc, half_court,
                      hc_big_circle, hc_sm_circle]

    # Add the court elements onto the axes
    for element in court_elements:
        ax.add_patch(element)

    return ax