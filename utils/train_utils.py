import os
import sys

import socket
import time
import argparse
import pickle
import datetime
import multiprocessing

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as torchF
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.multiprocessing as mp


def fetch_ckpt_namelist(ckptdir, suffix):
    ckpts = []
    for x in os.listdir(ckptdir):
        if x.endswith(suffix) and (not x.startswith('best_')):
            xs = x.replace(suffix, '')
            ckpts.append((x, int(xs)))
    if len(ckpts) == 0:
        return []
    else:
        ckpts.sort(key=lambda x: x[1])
        return ckpts


def get_last_ckpt(ckptdir, device, suffix='_checkpoint.pt', specify=None):
    if specify is not None:
        last_ckpt = torch.load(os.path.join(ckptdir, '{}'.format(specify) + suffix))
    else:
        ckpts = fetch_ckpt_namelist(ckptdir, suffix)
        if len(ckpts) == 0:
            last_ckpt = None
        else:
            last_ckpt = torch.load(os.path.join(ckptdir, ckpts[-1][0]), map_location=device)
    if os.path.exists(os.path.join(ckptdir, 'best' + suffix)):
        best_ckpt = torch.load(os.path.join(ckptdir, 'best' + suffix), map_location=device)
    else:
        best_ckpt = None
    return {
        'last': last_ckpt, 'best': best_ckpt
    }


def save_ckpt(epoch, best_valid_loss, best_valid_epoch, model, optimizer, scheduler, ckptdir,
              prefix, suffix='_checkpoint.pt', max_to_keep=3):
    ckptdict = {
        'epoch': epoch,
        'best_valid_loss': best_valid_loss,
        'best_valid_epoch': best_valid_epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(ckptdict, os.path.join(ckptdir, prefix + suffix))
    # remove too old ckpts
    ckpts = fetch_ckpt_namelist(ckptdir, suffix)
    if len(ckpts) > max_to_keep:
        for tdfname, _ in ckpts[:len(ckpts) - max_to_keep]:
            to_del_path = os.path.join(ckptdir, tdfname)
            os.remove(to_del_path)
    return ckptdict


def load_ckpt(model, optimizer, scheduler, ckpt, restore_opt_sche=True):
    epoch = ckpt['epoch']
    best_valid_loss = ckpt['best_valid_loss']
    best_valid_epoch = ckpt['best_valid_epoch']
    try:
        model.load_state_dict(ckpt['model'])
    except:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(ckpt['model'])
        model = model.module
    if restore_opt_sche:
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
    return epoch, best_valid_loss, best_valid_epoch, model, optimizer, scheduler
