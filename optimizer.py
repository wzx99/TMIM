import logging
import torch
import torch.nn as nn
import torch.distributed as dist
from utils.pyt_utils import get_main_flag


def build_optimizer(config, model):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    if config.get('set_lr_decay', None) is not None:
        key, ratio = config['set_lr_decay']
        lr = ratio*config['optim_kwargs']['lr']
        parameters = set_lr_decay(model, key, lr)
    else:
        parameters = model.parameters()

    if config['optim'] == 'sgd':
        from torch.optim import SGD as optimizer_class
    elif config['optim'] == 'adamw':
        from torch.optim import AdamW as optimizer_class
    optimizer = optimizer_class(parameters, **config['optim_kwargs'])
    optimizer.zero_grad()
    return optimizer


def set_lr_decay(model, key, lr):
    no_decay = []
    has_decay = []
    has_decay_names = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if key in name:
            has_decay.append(param)
            has_decay_names.append(name)
        else:
            no_decay.append(param)
    if len(has_decay) > 0 and get_main_flag():
        logging.info('**** some parameters has lr decay ****')
        # logging.info(has_decay_names)
    return [{'params': no_decay},
            {'params': has_decay, 'lr': lr}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


def build_scheduler(config, optimizer):
    if config['scheduler']=='OneCycleLR':
        from torch.optim.lr_scheduler import OneCycleLR as scheduler_class
    elif config['scheduler']=='MultiStepLR':
        from torch.optim.lr_scheduler import MultiStepLR as scheduler_class
    elif config['scheduler']=='LinearLR':
        from torch.optim.lr_scheduler import LinearLR as scheduler_class
    elif config['scheduler']=='CosineLRScheduler':
        from timm.scheduler.cosine_lr import CosineLRScheduler as scheduler_class
    else:
        return None
    scheduler = scheduler_class(optimizer, **config['scheduler_kwargs'])
    return scheduler
