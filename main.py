"""
Multiple Sequence Imputation with Equivariant Dual Graph Networks

Requirements:
    Python>=3.6.0
    PyTorch>=1.2.0
    dgl>=0.4

Usage:
    $ python main.py
"""
from __future__ import division
from __future__ import print_function

import sys
import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import logging
import pprint
# import socket
import datetime
import yaml
import json

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import dgl
import dgl.function as fn

from tqdm import tqdm

from model import RGN, SGN, SNATA
from baselines import LinearRegressionImputation, GRUImputation, KNNImputation, HHopImputation, ExtraTreesImputation
from utils.data.nba_utils import NBADataset, nba_collate_fn, load_nba_data
from utils.data.noaa_utils import NOAADataset, noaa_collate_fn, load_noaa_data
from utils.train_utils import save_ckpt, load_ckpt, get_last_ckpt


def main(cfg):

    # For reproduction
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.multiprocessing.set_sharing_strategy('file_system')

    # General setting
    MODEL = cfg['model']['MODEL']
    DATATYPE = cfg['dataset']['TYPE']
#     gameId = DATATYPE.split("-")[1]

    if cfg['run'] == 'newexp':
        exp_start_time = datetime.datetime.now().isoformat()
        dirname = os.path.join('artifacts', MODEL, DATATYPE, exp_start_time)
        cfg['dirname'] = dirname
    elif cfg['run'] == 'loadexp':
        dirname = cfg['dirname']
    else:
        raise NotImplementedError(cfg['run'])
    logdir = os.path.join(dirname, 'log')
    ckptdir = os.path.join(dirname, 'ckpt')
    for d in [logdir, ckptdir]:
        if not os.path.exists(d):
            os.makedirs(d)
    logfilename = os.path.join(logdir, 'log.txt')
    if cfg['run'] == 'newexp':
        with open(os.path.join(dirname, 'cfg.yaml'), 'w') as cfgf:
            yaml.dump(cfg, cfgf, default_flow_style=False)
    

    # Print the configuration - just to make sure that you loaded what you wanted to load
    with open(logfilename, 'a') as f:
        pp = pprint.PrettyPrinter(indent=4, stream=f)
        pp.pprint(cfg)

    import logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=logfilename,
                        filemode='a',
                        format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.DEBUG)
    logger = logging.getLogger()
    print(logger.handlers)
    writer = SummaryWriter(logdir)

    logger.info("MODEL: {}\tDATATYPE: {}".format(MODEL, DATATYPE))
    logger.info("logdir: {}".format(logdir))
    logger.info("ckptdir: {}".format(ckptdir))

    ########## Device setting ##########
    if cfg['gpu'] == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(cfg['gpu']) if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    ####################################

    node_unseen_ids = None
    if DATATYPE.startswith('nba'):
        datadict = load_nba_data(cfg)
        train_dataset = datadict['dataset']['train']
        valid_dataset = datadict['dataset']['valid']
        test_dataset = datadict['dataset']['test']
        batch_size = cfg['train']['batch_size']
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=nba_collate_fn)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=nba_collate_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8, collate_fn=nba_collate_fn)
        num_edge_types = datadict['meta']['num_edge_types']
    elif DATATYPE.startswith('noaa'):
        datadict = load_noaa_data(cfg)
        train_dataset = datadict['dataset']['train']
        valid_dataset = datadict['dataset']['valid']
        test_dataset = datadict['dataset']['test']
        batch_size = cfg['train']['batch_size']
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=noaa_collate_fn)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=noaa_collate_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8, collate_fn=noaa_collate_fn)
        num_edge_types = datadict['meta']['num_edge_types']
        if 'node_unseen_ids' in datadict:
            node_unseen_ids = datadict['node_unseen_ids']

    #### Model ####
    if MODEL == 'SNATA':
        ########## Architecture setting ##########
        edge_emb_dim = cfg['model']['edge_emb_dim']
        rgn_input_edge_dim = cfg['model']['rgn_input_edge_dim']
        rgn_input_node_dim = cfg['model']['rgn_input_node_dim']
        rgn_global_dim = cfg['model']['rgn_global_dim']
        rgn_hidden_dim = cfg['model']['rgn_hidden_dim']
        sgn_input_edge_dim = cfg['model']['sgn_input_edge_dim']
        sgn_input_node_dim = cfg['model']['sgn_input_node_dim']
        sgn_global_dim = cfg['model']['sgn_global_dim']
        sgn_hidden_dim = cfg['model']['sgn_hidden_dim']
        sgn_output_node_dim = cfg['model']['sgn_output_node_dim']
        num_spatial_hops = cfg['model']['num_spatial_hops']
        ##########################################
        rgn_model = RGN(input_edge_dim=rgn_input_edge_dim,
                        input_node_dim=rgn_input_node_dim,
                        rgn_global_dim=rgn_global_dim,
                        rgn_hidden_dim=rgn_hidden_dim,
                        num_spatial_hops=num_spatial_hops,
                        device=device)
        rgn_model.to(device)

        num_total_params = sum(p.numel() for p in rgn_model.parameters() if p.requires_grad)
        logger.info("# params in RGN: {}".format(num_total_params))


        sgn_model = SGN(input_edge_dim=sgn_input_edge_dim,
                        input_node_dim=sgn_input_node_dim,
                        sgn_global_dim=sgn_global_dim,
                        sgn_hidden_dim=sgn_hidden_dim,
                        rgn_hidden_dim=rgn_hidden_dim,
                        output_node_dim=sgn_output_node_dim)
        sgn_model.to(device)

        num_total_params = sum(p.numel() for p in sgn_model.parameters() if p.requires_grad)
        logger.info("# params in SGN: {}".format(num_total_params))

        model = SNATA(rgn_model, sgn_model, edge_embedding=nn.Embedding(num_edge_types, edge_emb_dim), device=device)
        model.to(device)
    elif MODEL == 'LinearRegression':
        model = LinearRegressionImputation(cfg['model']['input_node_dim'], cfg['model']['output_node_dim'])
        model.to(device)
    elif MODEL == 'GRU':
        model = GRUImputation(cfg['model']['input_node_dim'], cfg['model']['output_node_dim'],
            cfg['model']['hidden_dim'], cfg['model']['layer_num'])
        model.to(device)
    elif MODEL == 'KNN':
        model = KNNImputation(cfg['model']['input_node_dim'], cfg['model']['output_node_dim'])
        model.to(device)
    elif MODEL == 'HHop':
        model = HHopImputation(cfg['model']['input_node_dim'], cfg['model']['output_node_dim'], h_hop=1)
        model.to(device)
    elif MODEL == 'ExtraTrees':
        model = ExtraTreesImputation(cfg['model']['input_node_dim'], cfg['model']['output_node_dim'])
        model.to(device)
    else:
        raise NotImplementedError('Model {} not implemented!'.format(MODEL))

    num_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("# params in model: {}".format(num_total_params))
    print("# params in model: {}".format(num_total_params))

    #### Training and Optimizer setting
    loss_fn = nn.MSELoss()
    input_seq_length = cfg['train']['input_seq_length']
    num_epochs = cfg['train']['num_epochs']
    val_check_step = cfg['train']['val_check_step']
    try:
        test_check_step = cfg['train']['test_check_step']
    except:
        test_check_step = num_epochs - 1
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=cfg['optimizer']['initial_lr'],
                           weight_decay=cfg['optimizer']['weight_decay'])
    filled_rgn = cfg['train']['filled_rgn']
    reducer = cfg['optimizer']['reducer']
    # scheduler (not used for now)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500)

    start_epoch = 0
    min_val_epoch = 0
    min_val_loss = np.float('inf')
    if cfg['run'] == 'newexp':
        logger.info("new model is initialized. {}".format(ckptdir))
    elif cfg['run'] == 'loadexp':
        loaded_last_ckpt = get_last_ckpt(ckptdir, 'cpu')
        last_ckpt = loaded_last_ckpt['last']
        if last_ckpt is not None:
            start_epoch, min_val_loss, min_val_epoch, model, optimizer, scheduler = load_ckpt(model, optimizer, scheduler, last_ckpt)
            logger.info("loaded ckpt. {}".format(ckptdir))
        else:
            logger.info("new model is initialized. {}".format(ckptdir))
    else:
        raise NotImplementedError(cfg['run'])

    last_save_time = datetime.datetime.now()
    for _epoch in range(start_epoch, num_epochs):
        if cfg['mode'] == 'train':
            model.train()
            tr_losses_events = []
            
            use_self_supervision = True
            if 'use_self_supervision' in cfg['train'] and cfg['train']['use_self_supervision'] is False:
                print('Not using self supervision for training!')
                use_self_supervision = False

            for rgn_graph_input_batch_list, sgn_graph_input_batch, input_X_seq_batch, mask_batch in tqdm(train_dataloader, total=len(train_dataloader)):
                rgn_graph_input_batch_list = [x.to(device) for x in rgn_graph_input_batch_list]
                sgn_graph_input_batch = sgn_graph_input_batch.to(device)
                input_X_seq_batch = input_X_seq_batch.to(device)
                mask_batch = mask_batch.to(device)
                out_graph, _ = model(rgn_graph_input_batch_list, sgn_graph_input_batch)
                pred = out_graph.ndata['h_v']
                if use_self_supervision:
                    subseqnum = input_X_seq_batch.shape[0]
                    loss = loss_fn(pred, input_X_seq_batch.flatten(0, 1)) * subseqnum
                    if reducer == "MEAN":
                        event_loss = loss / subseqnum
                    else:
                        event_loss = loss / train_dataloader.batch_size
                else:
                    loss_filter = input_X_seq_batch.new_zeros(input_X_seq_batch.shape[0], input_X_seq_batch.shape[1]).float()
                    for tt in range(mask_batch.shape[0]):
                        temp = loss_filter[tt]
                        temp[mask_batch[tt]] = 1.0
                        loss_filter[tt] = temp
                    loss_filter = loss_filter.reshape(-1, 1)

                    subseqnum = input_X_seq_batch.shape[0]
                    loss = nn.MSELoss(reduction='sum')(pred * loss_filter, input_X_seq_batch.flatten(0, 1) * loss_filter) / torch.sum(loss_filter) * subseqnum
                    if reducer == "MEAN":
                        event_loss = loss / subseqnum
                    else:
                        event_loss = loss / train_dataloader.batch_size
                optimizer.zero_grad()
                try:
                    event_loss.backward()
                    optimizer.step()
                except RuntimeError:
                    print('no autograd needed!')
                tr_losses_events.append(event_loss.item())
                # print(tr_losses_events[-1])

            writer.add_scalars('loss/train', {'loss': np.mean(tr_losses_events)}, _epoch)
            logger.info("[Train] Average training error (epoch {}): {}\t over {} events"
                        .format(_epoch, np.mean(tr_losses_events), len(tr_losses_events)))

            if _epoch%val_check_step == 0:
                model.eval()
                with torch.no_grad():
                    val_losses = []
                    for rgn_graph_input_batch_list, sgn_graph_input_batch, input_X_seq_batch, mask_batch in tqdm(valid_dataloader, total=len(valid_dataloader)):
                        subseqnum = input_X_seq_batch.shape[0]
                        rgn_graph_input_batch_list = [x.to(device) for x in rgn_graph_input_batch_list]
                        sgn_graph_input_batch = sgn_graph_input_batch.to(device)
                        input_X_seq_batch = input_X_seq_batch.to(device)
                        out_graph, _ = model(rgn_graph_input_batch_list, sgn_graph_input_batch)
                        out_graph_list = dgl.unbatch(out_graph)
                        val_preds = [og.ndata['h_v'] for og in out_graph_list]
                        val_loss_list = [loss_fn(val_pred[mask], input_X_seq[mask]).item() for val_pred, input_X_seq, mask in zip(val_preds, input_X_seq_batch, mask_batch)]
                        val_losses.append(np.mean(val_loss_list))
                    val_loss = np.mean(val_losses)
                    writer.add_scalars('loss/valid', {'loss': val_loss}, _epoch)
                    if val_loss < min_val_loss:
                        min_val_loss = val_loss
                        min_val_epoch = _epoch
                        save_ckpt(_epoch, min_val_loss, min_val_epoch, model, optimizer, scheduler, ckptdir, prefix='best')
                        logger.info("[Valid] MSE at {}/{}\t[Train]: {:.4e}\t[Valid]: {:.4e}\t# val_losses: {} [Saved]"
                                    .format(_epoch+1, num_epochs, loss.item()/subseqnum, val_loss, len(val_losses)))
                    else:
                        logger.info("[Valid] MSE at {}/{}\t[Train]: {:.4e}\t[Valid]: {:.4e}\t# val_losses: {}"
                                    .format(_epoch+1, num_epochs, loss.item()/subseqnum, val_loss, len(val_losses)))

            save_time = datetime.datetime.now()
            save_intv = (save_time - last_save_time).total_seconds()
            # if save_intv > 300: # save every 5 min
            if True:
                last_save_time = save_time
                save_ckpt(_epoch, min_val_loss, min_val_epoch, model, optimizer, scheduler, ckptdir, prefix='{:d}'.format(_epoch))

        if (_epoch == num_epochs - 1) or (_epoch%test_check_step == 0):
            loaded_last_ckpt = get_last_ckpt(ckptdir, 'cpu')
            best_ckpt = loaded_last_ckpt['best']
            if type(model) in [LinearRegressionImputation, KNNImputation, HHopImputation, ExtraTreesImputation]:
                pass
            else:
                if best_ckpt is None:
                    raise NotImplementedError('No best ckpt yet! Train more!')
                _, min_val_loss, min_val_epoch, model, optimizer, scheduler = load_ckpt(model, optimizer, scheduler, best_ckpt)
            
            model.eval()
            with torch.no_grad():
                te_losses = []
                te_node_unseen_losses = []
                te_node_unseen_and_masked_losses = []
                test_results = []
                for rgn_graph_input_batch_list, sgn_graph_input_batch, input_X_seq_batch, mask_batch in tqdm(test_dataloader, total=len(test_dataloader)):
                    subseqnum = input_X_seq_batch.shape[0]
                    rgn_graph_input_batch_list = [x.to(device) for x in rgn_graph_input_batch_list]
                    sgn_graph_input_batch = sgn_graph_input_batch.to(device)
                    input_X_seq_batch = input_X_seq_batch.to(device)
                    if cfg['multistep_imputation_test']:
                        assert test_dataloader.batch_size == 1
                        multistep_imp_value = []
                        rgn_graph_inputs_list = [dgl.unbatch(x) for x in rgn_graph_input_batch_list]
                        sgn_graph_inputs = dgl.unbatch(sgn_graph_input_batch)
                        seqnum = len(sgn_graph_inputs)
                        te_loss = []
                        for seqi in range(seqnum):
                            rgn_graph_input_list = [dgl.batch([x[seqi],]) for x in rgn_graph_inputs_list]
                            sgn_graph_input = dgl.batch([sgn_graph_inputs[seqi],])
                            mask = mask_batch[seqi]
                            if len(multistep_imp_value) > 0:
                                last_imp_value = multistep_imp_value[-1]
                                sgn_graph_input.ndata['x'][mask, :last_imp_value.shape[1]] = last_imp_value[mask]
                                if cfg['test_imputed_as_true']:
                                    sgn_graph_input.ndata['x'][mask, last_imp_value.shape[1]:] = torch.tensor([1, 0, 0]).float().to(device)
                            out_graph, _ = model(rgn_graph_input_list, sgn_graph_input)
                            multistep_imp_value.append(out_graph.ndata['h_v'])
                            te_loss.append(loss_fn(multistep_imp_value[-1][mask], input_X_seq_batch[seqi][mask]).item())
                        multistep_imp_value = torch.stack(multistep_imp_value, dim=0)
                        te_loss = np.mean(te_loss)
                        te_losses.append(te_loss)
                    else:
                        # Refilling and Reprediction
                        threshold = cfg['test']['threshold']
                        prev_loss = np.float('inf')
                        prev_loss_node_unseen = 0
                        prev_loss_node_unseen_and_masked = 0
                        for ii in range(cfg['test']['num_refill_try']):
                            out_graph, _ = model(rgn_graph_input_batch_list, sgn_graph_input_batch)
                            out_graph_list = dgl.unbatch(out_graph)
                            te_preds = [og.ndata['h_v'] for og in out_graph_list]
                            te_loss_list = [loss_fn(te_pred[mask], input_X_seq[mask]).item() for te_pred, input_X_seq, mask in zip(te_preds, input_X_seq_batch, mask_batch)]
                            te_loss_ii = np.mean(te_loss_list)
                            if node_unseen_ids is not None:
                                node_unseen_loss_list = [loss_fn(te_pred[node_unseen_ids], input_X_seq[node_unseen_ids]).item() for te_pred, input_X_seq in zip(te_preds, input_X_seq_batch)]
                                node_unseen_loss_ii = np.mean(node_unseen_loss_list)
                                node_unseen_and_masked_loss_list = []
                                for te_pred, input_X_seq, mask in zip(te_preds, input_X_seq_batch, mask_batch):
                                    node_unseen_and_masked_ids = torch.from_numpy(np.intersect1d(node_unseen_ids,
                                        mask.data.cpu().numpy())).long().contiguous()
                                    node_unseen_and_masked_loss_list.append(loss_fn(te_pred[node_unseen_and_masked_ids], input_X_seq[node_unseen_and_masked_ids]).item())
                                node_unseen_and_masked_loss_ii = node_unseen_and_masked_loss_list
                                print(node_unseen_and_masked_ids)
                            if prev_loss >= te_loss_ii and prev_loss - te_loss_ii < threshold:
                                logger.info('converge after {} iterations'.format(ii))
                                if cfg['save_results']:
                                    prev_te_preds_arr = torch.stack(prev_te_preds, dim=0).data.cpu().numpy()
                                    input_X_seq_batch_arr = input_X_seq_batch.data.cpu().numpy()
                                    mask_batch_arr = mask_batch.data.cpu().numpy()
                                    test_results.append((prev_te_preds_arr, input_X_seq_batch_arr, mask_batch_arr))
                                break
                            if prev_loss < te_loss_ii:
                                logger.info('getting worse after {} iterations'.format(ii))
                                if cfg['save_results']:
                                    prev_te_preds_arr = torch.stack(prev_te_preds, dim=0).data.cpu().numpy()
                                    input_X_seq_batch_arr = input_X_seq_batch.data.cpu().numpy()
                                    mask_batch_arr = mask_batch.data.cpu().numpy()
                                    test_results.append((prev_te_preds_arr, input_X_seq_batch_arr, mask_batch_arr))
                                break
                            prev_loss = te_loss_ii
                            prev_te_preds = te_preds
                            if node_unseen_ids is not None:
                                prev_loss_node_unseen = node_unseen_loss_ii
                                prev_loss_node_unseen_and_masked = node_unseen_and_masked_loss_ii
                            filled_x = []
                            for te_pred, out_graph_i, mask in zip(te_preds, out_graph_list, mask_batch):
                                out_graph_i.ndata['x'][mask, :te_pred.shape[1]] = te_pred[mask]
                                if cfg['test_imputed_as_true']:
                                    out_graph_i.ndata['x'][mask, te_pred.shape[1]:] = torch.tensor([1, 0, 0]).float().to(device)
                                filled_x.append(out_graph_i.ndata['x'])
                            sgn_graph_input_batch.ndata['x'] = torch.cat(filled_x, dim=0)
                        te_losses.append(prev_loss)
                        if node_unseen_ids is not None:
                            te_node_unseen_losses.append(prev_loss_node_unseen)
                            te_node_unseen_and_masked_losses.extend(prev_loss_node_unseen_and_masked)
                te_losses = np.array(te_losses, dtype=np.float32)
                te_losses = te_losses[~np.isnan(te_losses)]
                if node_unseen_ids is not None:
                    te_node_unseen_losses = np.array(te_node_unseen_losses, dtype=np.float32)
                    # te_node_unseen_losses = te_node_unseen_losses[~np.isnan(te_node_unseen_losses)]
                    te_node_unseen_losses[np.isnan(te_node_unseen_losses)] = 0
                    logger.info("[TEST] # te_losses: {}\t Average of te_losses: {:.4e}. Average of te_node_unseen_losses: {:.4e}. Average of te_node_unseen_and_masked_losses: {:.4e}.".format(
                        len(te_losses), np.mean(te_losses), np.mean(te_node_unseen_losses), np.nanmean(te_node_unseen_and_masked_losses)))
                    writer.add_scalars('loss/test', {'loss': np.mean(te_losses), 'node_unseen_loss': np.mean(te_node_unseen_losses), 'node_unseen_and_masked_loss':  np.nanmean(te_node_unseen_and_masked_losses)}, _epoch)
                    print(np.mean(te_losses), np.mean(te_node_unseen_losses),  np.nanmean(te_node_unseen_and_masked_losses))
                else:
                    logger.info("[TEST] # te_losses: {}\t Average of te_losses: {:.4e}".format(len(te_losses), np.mean(te_losses)))
                    writer.add_scalars('loss/test', {'loss': np.mean(te_losses)}, _epoch)
                    print(np.mean(te_losses))

                if cfg['save_results']:
                    test_results = list(zip(*test_results))
                    test_results = [np.concatenate(x, axis=0) for x in test_results]
                    np.savez(os.path.join(logdir, 'test_results.npz'), te_preds=test_results[0], input_X_seq_batch=test_results[1], mask_batch=test_results[2])

                if cfg['mode'] == 'test':
                    break

            loaded_last_ckpt = get_last_ckpt(ckptdir, 'cpu')
            last_ckpt = loaded_last_ckpt['last']
            if type(model) in [LinearRegressionImputation, KNNImputation, HHopImputation]:
                pass
            else:
                if last_ckpt is None:
                    raise NotImplementedError('No last ckpt yet!')
                _, min_val_loss, min_val_epoch, model, optimizer, scheduler = load_ckpt(model, optimizer, scheduler, best_ckpt)


def load_cfg(yaml_filepath):
    """
    Load a YAML configuration file.

    Parameters
    ----------
    yaml_filepath : str

    Returns
    -------
    cfg : dict
    """
    # Read YAML experiment definition file
    with open(yaml_filepath, 'r') as stream:
        cfg = yaml.load(stream, Loader=yaml.FullLoader)
#     cfg = make_paths_absolute(os.path.dirname(yaml_filepath), cfg)
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    cfg = make_paths_absolute(ROOT_DIR, cfg)
    return cfg

def make_paths_absolute(dir_, cfg):
    """
    Make all values for keys ending with `_path` absolute to dir_.

    Parameters
    ----------
    dir_ : str
    cfg : dict

    Returns
    -------
    cfg : dict
    """
    for key in cfg.keys():
        if key.endswith("_path"):
            cfg[key] = os.path.join(dir_, cfg[key])
            cfg[key] = os.path.abspath(cfg[key])
            if not os.path.isfile(cfg[key]):
                logging.error("%s does not exist.", cfg[key])
        if type(cfg[key]) is dict:
            cfg[key] = make_paths_absolute(dir_, cfg[key])
    return cfg

def get_parser():
    """Get parser object."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Implementation of Spatially Non-Autoregressive Temporally Autoregressive Sequence Imputation',
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("-f", "--file",
                        dest="filename",
                        help="experiment definition file (YAML format)",
                        metavar="FILE",
                        default='')
    parser.add_argument("--savedexp", help="load conf and ckpt from saved exp", default='')
    parser.add_argument("--gpu",
                        type=int,
                        default=0,
                        help="gpu number: 0 or 1")
    parser.add_argument("--mode", type=str, default="train", help="train or test")
    parser.add_argument("--multistep_imputation_test", action='store_true')
    parser.add_argument("--test_imputed_as_true", action='store_true')
    parser.add_argument("--save_results", action='store_true')
    # parser.add_argument("--seed",
    #                     type=int,
    #                     default=42,
    #                     help="random seed")

    return parser


if __name__=="__main__":
    args = get_parser().parse_args()

    if len(args.filename) > 0:
        # new experiment
        cfg = load_cfg(args.filename)
        cfg['run'] = 'newexp'
    elif len(args.savedexp) > 0:
        # load experiment
        cfgfile = os.path.join(args.savedexp, 'cfg.yaml')
        cfg = load_cfg(cfgfile)
        cfg['run'] = 'loadexp'
    else:
        raise NotImplementedError('No configuration loaded!')

#    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # cfg['seed'] = args.seed
    cfg['gpu'] = args.gpu
    cfg['mode'] = args.mode
    cfg['multistep_imputation_test'] = args.multistep_imputation_test
    cfg['test_imputed_as_true'] = args.test_imputed_as_true
    cfg['save_results'] = args.save_results

    main(cfg)
