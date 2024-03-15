import math
import os
import subprocess
import sys
import time
import argparse
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.transforms.functional as visionF
import torch.cuda.amp as amp
from typing import Optional
import matplotlib.pyplot as plt

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter
from timm.utils import ModelEma as ModelEma
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from configs import *
from models import *
# from loss import *
from data import *
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import *
#denormalize, load_checkpoint, load_checkpoint_finetune, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


scaler = amp.GradScaler()
logger = None

def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--data',type=str,default='two_plas',help='dataset')
    # parser.add_argument('--cfg', type=str, default="/data/hkt/work_microplastic/res_50_two/configs/microplas/res50_two.yaml" ,metavar="FILE", help='path to config file', )#required=True,
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, default=32, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, default='/data/hkt/work_microplastic/data_pre/sam_six_other3', help='path to dataset')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--finetune', help='finetune from checkpoint')

    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")

    parser.add_argument('--output', default='/data/hkt/work_microplastic/result/', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', default=False,action='store_true', help='Perform evaluation only')

    # ema
    parser.add_argument('--model-ema',default=False, action='store_true')


    # distributed training
    parser.add_argument("--local_rank", type=int, required=False, help='local rank for DistributedDataParallel')

    parser.add_argument('--dist-url', default='tcp://127.0.0.1:8090', type=str,
                    help='url used to set up distributed training')

    args, unparsed = parser.parse_known_args()
    config = select_config(args)

    return args, config


def main(config):
    config.defrost()
    # print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    print("-------------------")

    # linear scale the learning rate according to total batch size, base bs 1024
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE/ 256.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE / 256.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE  / 256.0

    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.train_reconstrution = os.path.join(config.OUTPUT, "train/reconstrution")
    if not os.path.exists(config.train_reconstrution):
        os.makedirs(config.train_reconstrution)
    config.val_reconstrution = os.path.join(config.OUTPUT, "val/reconstrution")
    if not os.path.exists(config.val_reconstrution):
        os.makedirs(config.val_reconstrution)
    # config.LOCAL_RANK = rank
    config.freeze()


    seed = config.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    global logger

    logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")
    logger.info(config.dump())
    writer = None
    writer = SummaryWriter(config.OUTPUT)

    dataset_train, dataset_val, data_loader_train, data_loader_val = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)

    model.cuda()
    print_peak_memory("Max memory allocated after creating local model", config.LOCAL_RANK)
    # logger.info(str(model)[:10000])
    logger.info(str(model))

    print_peak_memory("Max memory allocated after creating DDP", config.LOCAL_RANK)
        
    optimizer = build_optimizer(config, model)
    if config.TRAIN.AMP:
        logger.info(f"-------------------------------Using Pytorch AMP...--------------------------------")



    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")

    lr_scheduler = build_scheduler(config)



    criterion_ce =torch.nn.CrossEntropyLoss()
    criterion_mse = torch.nn.MSELoss()


    max_accuracy = 0.0


    logger.info("******************Start training*********************")
    start_time = time.time()

    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        # data_loader_train.set_epoch(epoch)
        train_one_epoch(config, model, criterion_ce,criterion_mse, data_loader_train, optimizer, epoch,  lr_scheduler, writer, model_ema)


        # acc1, 
        acc1,_ = validate(config, data_loader_val, model, writer, epoch)



        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy}%')

        if epoch % config.SAVE_FREQ == 0:
            save_checkpoint(config, epoch, model, acc1, max_accuracy, optimizer, logger, model_ema)


        

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

    exit(0)


def train_one_epoch(config, model, criterion_ce,c_mse, data_loader, optimizer, epoch,lr_scheduler, writer, model_ema: Optional[ModelEma] = None):
    global logger
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    fctloss_meter = AverageMeter()
    fcbloss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    data_time = AverageMeter()

    start = time.time()
    end = time.time()


    for idx, (samples, targets,_) in enumerate(data_loader):

        samples = samples.cuda()
        targets = targets.cuda()
        targetsre = targets.reshape(-1,1).float()
        
        data_time.update(time.time()-end)
        optimizer.zero_grad()
        lr_scheduler.step_update(optimizer, idx / num_steps + epoch, config)

        output,msel,fct,fcb,tloss,bloss= model(samples)

        mseloss = c_mse(output, samples)
        fctloss = criterion_ce(fct, targets)
        fcbloss = criterion_ce(fcb, targets)
        loss = mseloss + msel+fctloss+fcbloss+tloss+bloss
        acc1 = accuracy(fct, targets, topk=(1,))
        acc1 = torch.tensor(acc1)
        acc1_meter.update(acc1.item(), targets.size(0))

        if not math.isfinite(loss.item()):
            print("Loss is {} in iteration {} !".format(loss.item(), idx))

        loss.backward()
        optimizer.step()
        if config.TRAIN.CLIP_GRAD:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)


        loss_meter.update(loss.item(), targets.size(0))
        fctloss_meter.update(fctloss.item(), targets.size(0))
        fcbloss_meter.update(fcbloss.item(), targets.size(0))


        # norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx%10 == 0:
            writer.add_scalar('Train/train_loss',loss_meter.val, epoch * num_steps + idx )
            writer.add_scalar('Train/train_mseloss',mseloss.item(), epoch * num_steps + idx )
            writer.add_scalar('Train/train_fctloss',fctloss_meter.val, epoch * num_steps + idx )
            writer.add_scalar('Train/train_fcbloss',fcbloss_meter.val, epoch * num_steps + idx )
            writer.add_scalar('Train/train_acc',acc1_meter.val, epoch * num_steps + idx )


        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[-1]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'datatime {data_time.val:.4f} ({data_time.avg:.4f})\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'acc1 {acc1_meter.val:.4f}\t'
                f'mem {memory_used:.0f}MB')

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model, writer, epoch):
    global logger
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    tacc_meter = AverageMeter()

 
    end = time.time()
    for idx, (images, target,_) in enumerate(data_loader):

        images = images.cuda()
        target = target.cuda()
        targetre = target.reshape(-1,1).float()
        
        # compute output
        output,msel,fct,fcb,tloss,bloss = model(images)



        mseloss = mse(output, images)
        fctloss = criterion(fct, target)
        fcbloss = criterion(fcb, target)
        loss = mseloss+msel+fctloss+fcbloss+tloss+bloss
        loss_meter.update(loss.item(), target.size(0))




        acc1= accuracy(fct, target, topk=(1,))
        acc1 = torch.tensor(acc1)
        tacc_meter.update(acc1.item(), target.size(0))

        batch_time.update(time.time() - end)
        end = time.time()


        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc1 {tacc_meter.val:.3f} ({tacc_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')



    return acc1,loss_meter.avg




if __name__ == "__main__":
    _,config = parse_option()

    cudnn.benchmark = True

    os.makedirs(config.OUTPUT, exist_ok=True)
    main(config)
