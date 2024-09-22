import sys
import os
import argparse
from tqdm import tqdm
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils import data

# import warnings
# warnings.filterwarnings("ignore")

from engine import Engine
from models import build_model
from datasets import build_dataset
from optimizer import build_optimizer, build_scheduler
from utils.pyt_utils import load_model, all_reduce_tensor, get_main_flag
from utils.logger import get_logger
from mmengine.config import Config


def get_parser():
    parser = argparse.ArgumentParser(description="recognition")
    parser.add_argument("--cfg", type=str, default=None)
    parser.add_argument("--ckpt-name", type=str, default='ckpt')
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--start-iter", type=int, default=0)
    parser.add_argument("--save-log", action="store_true")
    return parser


def main():
    parser = get_parser()

    with Engine(custom_parser=parser) as engine:
        main_flag = get_main_flag()
        
        args = parser.parse_args()
        cfg = Config.fromfile(args.cfg)
        cfg.snapshot_dir = os.path.join(cfg.snapshot_dir, args.ckpt_name)
        
        if not os.path.exists(cfg.snapshot_dir):
            if main_flag:
                os.makedirs(cfg.snapshot_dir)
        time.sleep(1)
                
        if args.save_log:
            logger = get_logger(log_file=os.path.join(cfg.snapshot_dir, 'log.txt'))
        else:
            logger = get_logger()

        if main_flag:
            logger.info('Start Train')
            logger.info('Running with argument: \n{}'.format(
                '\n'.join('{}:{}'.format(k,v) for k,v in vars(args).items())))
            logger.info('Running with config: \n{}'.format(cfg._text))
                
        cudnn.benchmark = True
        seed = cfg.random_seed
        if engine.distributed:
            seed = cfg.random_seed + engine.local_rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # data loader
        train_dataset = build_dataset(cfg.train_data_cfg)
        train_loader, train_sampler = engine.get_train_loader(train_dataset, cfg.train_data_cfg)
        val_dataset = build_dataset(cfg.val_data_cfg, 'val')
        val_loader, val_sampler = engine.get_test_loader(val_dataset, cfg.val_data_cfg)
        if main_flag:
            logger.info('Total {} samples for training'.format(train_dataset.__len__()))

        # model
        model = build_model(cfg.model_cfg)

        if args.resume:
            load_model(model, args.resume,ignore_prefix='module.')
            if main_flag:
                logger.info('resume from {}'.format(args.resume))

        optimizer = build_optimizer(cfg.optim_cfg, model)
        scheduler = build_scheduler(cfg.optim_cfg, optimizer)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        model = engine.data_parallel(model)
        model.train()
            
        run = True
        global_iteration = args.start_iter
        best_accuracy = -100
        while run:
            epoch = global_iteration // len(train_loader)
            if engine.distributed:
                train_sampler.set_epoch(epoch)

            for idx, data in enumerate(train_loader):
                global_iteration += 1

                optimizer.zero_grad()

                loss = model(data, global_iteration, 'train')
                reduce_loss = all_reduce_tensor(loss)
                
                loss.backward() #scaler.scale(loss).backward()
                optimizer.step() #scaler.step(optimizer)

                if global_iteration % cfg.log_interval == 0 and main_flag:
                    print_str = 'Epoch{}/Iters{}'.format(epoch, global_iteration) \
                            + ' Iter{}/{}:'.format(idx + 1, len(train_loader)) \
                            + ' lr=%.2e' % optimizer.param_groups[0]['lr']\
                            + ' loss=%.2f' % reduce_loss.item()
                    logger.info(print_str)
                
                # validation part
                if (global_iteration % cfg.val_interval == 0 or global_iteration == 1 or global_iteration >= cfg.total_iter):
                    if cfg.val:
                        model.eval()
                        with torch.no_grad():
                            outputs = []
                            for val_iter, batch in enumerate(val_loader):
                                outputs.append(model(batch, val_iter, 'val'))
                            if engine.distributed:
                                val_acc, val_dict = model.module.validation_epoch_end(outputs)
                            else:
                                val_acc, val_dict = model.validation_epoch_end(outputs)
                        model.train()
                    
                        if main_flag:
                            logger.info('*******************')
                            logger.info('Validation: ')
                            print_str = 'Epoch{}/Iters{}'.format(epoch, global_iteration) \
                                    + ' lr=%.2e' % optimizer.param_groups[0]['lr']
                            for k,v in val_dict.items():
                                print_str += ' %s=%.4f' % (k,v)
                            logger.info(print_str)

                            if val_acc > best_accuracy:
                                best_accuracy = val_acc
                                torch.save(model.state_dict(), os.path.join(cfg.snapshot_dir,'best_accuracy.pth'))
                                logger.info('best_acc=%.2f' % val_acc.item())
                            logger.info('*******************')

                    if main_flag:
                        logger.info('taking snapshot ...')
                        torch.save(model.state_dict(), os.path.join(cfg.snapshot_dir, 'latest.pth'))

                        if global_iteration >= cfg.start_save_iter:
                            torch.save(model.state_dict(), os.path.join(cfg.snapshot_dir, 'iter_'+str(global_iteration) + '.pth'))

                if global_iteration >= cfg.total_iter:
                    run = False
                    break

                if scheduler is not None:
                    scheduler.step(global_iteration)

if __name__ == '__main__':
    main()
