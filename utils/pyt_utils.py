# encoding: utf-8
import os
import sys
import time
import argparse
import logging
from collections import OrderedDict, defaultdict

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.distributed as dist


def get_main_flag():
    return (not dist.is_initialized()) or (dist.is_initialized() and dist.get_rank()==0)

def all_reduce_tensor(tensor, op=dist.ReduceOp.SUM, norm=True):
    if dist.is_initialized():
        tensor = tensor.clone()
        dist.all_reduce(tensor, op)
        if norm:
            tensor.div_(dist.get_world_size())

    return tensor


def load_model(model, model_file, is_restore=False, ignore_prefix=None, extra_prefix=None):
    t_start = time.time()
    if isinstance(model_file, str):
        device = torch.device('cpu')
        state_dict = torch.load(model_file, map_location=device)
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict.keys():
            state_dict = state_dict['state_dict']
    else:
        state_dict = model_file
    t_ioend = time.time()
    
    if ignore_prefix is not None:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k if not k.startswith(ignore_prefix) else k[len(ignore_prefix):]
            new_state_dict[name] = v
        state_dict = new_state_dict

    if extra_prefix is not None:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = extra_prefix + k
            new_state_dict[name] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict, strict=False)
    ckpt_keys = set(state_dict.keys())
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    unexpected_keys = ckpt_keys - own_keys
    
    missing_keys_list = []
    for k in missing_keys:
        if not k.endswith('.num_batches_tracked'):
            missing_keys_list.append(k)
    if len(missing_keys_list) > 0 and get_main_flag():
        print('Missing key(s) in state_dict: {}'.format(
            ', '.join('{}'.format(k) for k in missing_keys_list)))
        logging.warning('Missing key(s) in state_dict')
    
    unexpected_keys_list = []
    for k in unexpected_keys:
        if not k.endswith('.num_batches_tracked'):
            unexpected_keys_list.append(k)
    if len(unexpected_keys_list) > 0 and get_main_flag():
        print('Unexpected key(s) in state_dict: {}'.format(
            ', '.join('{}'.format(k) for k in unexpected_keys_list)))
        logging.warning('Unexpected key(s) in state_dict')

    del state_dict
    t_end = time.time()
    if get_main_flag():
        logging.info(
            "Load model, Time usage:\n\tIO: {}, initialize parameters: {}".format(
                t_ioend - t_start, t_end - t_ioend))

#     return model


def extant_file(x):
    """
    'Type' for argparse - checks that file exists but does not open.
    """
    if not os.path.exists(x):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(x))
    return x

