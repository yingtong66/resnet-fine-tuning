# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import time
import datetime
import random
import numpy as np

from datetime import timedelta

import torch
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models

# from models.modeling import VisionTransformer, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader
from utils.dist_util import *
from timm.utils import accuracy, AverageMeter


def parse_option():
    parser = argparse.ArgumentParser('Resnet50 training and evaluation script', add_help=False)
    # Required parameters
    parser.add_argument("--name", required=True, type=str, help="Name of this run. Used for monitoring.")
    parser.add_argument('--save_dir', default='./logs', type=str)
    parser.add_argument('--device_name', type=str, default='torch.cuda.get_device_name(0)')
    # data
    parser.add_argument("--data", default="rubbish", help="Which downstream task.")
    parser.add_argument('--data_path', type=str, default="./enhanced_data")
    parser.add_argument('--num_classes', type=int, default=40)
    # train
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--lr", default=3e-2, type=float, help="The initial learning rate for SGD.")
    parser.add_argument("--max_accuracy", default=0.0, type=float)
    # model
    parser.add_argument("--pretrain_dir", type=str, default="./pretrain", help="Where to search for pretrained ViT models.")
    parser.add_argument('--pretrain', type=str, default="ViT-B_16.npz", help='vit_base_patch16_224_in21k.pth')
    parser.add_argument('--model_file', type=str, default='modeling')
    # 是否冻结权重

    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    args = parser.parse_args()
    return args


def setup(args, logger):
    loc = 'cuda:{}'.format(0)
    checkpoint = torch.load('checkpoint_0099.pth.tar', map_location=loc)

    # create new OrderedDict that does only contain module.encoder_k statedict
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        if k[:17]=='module.encoder_k.':
            name = k[17:] # remove 'module.encoder_k.'
            new_state_dict[name] = v
    resnet50 = models.resnet50(num_classes=128)
    dim_mlp = resnet50.fc.weight.shape[1]
    print(dim_mlp)
    resnet50.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), resnet50.fc)
    resnet50.load_state_dict(new_state_dict)

    # resnetayt-20epoch模型
    resnet50.fc = nn.Linear(dim_mlp, 40)  # 将输出类别数设置为40
    # resnetayt-20epoch-2linear模型
    # resnet50.fc=nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.Dropout(p=0.5), nn.ReLU(), nn.Linear(dim_mlp, 40))
    
    resnet50.to(args.device)
    
    logger.info("Training parameters %s", args)
    return args, resnet50


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def main(args):
    # get logger
    creat_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())  # 获取训练创建时间
    args.path_log = os.path.join(args.save_dir, f'{args.data}', f'{args.name}')  # 确定训练log保存路径
    os.makedirs(args.path_log, exist_ok=True)  # 创建训练log保存路径
    logger = create_logging(os.path.join(args.path_log, '%s-%s-train.log' % (creat_time, args.name)))  # 创建训练保存log文件
    
    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, False, False))

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args, logger)

    # Prepare data
    train_loader, val_loader, test_loader = get_loader(args)

    # print args
    for param in sorted(vars(args).keys()):  # 遍历args的属性对象
        logger.info('--{0} {1}'.format(param, vars(args)[param]))

    # Training
    """ Train the model """
    tb_writer = SummaryWriter(log_dir=args.path_log)

    # get optimizer and scheduler
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=len(train_loader) * args.epochs)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=len(train_loader) * args.epochs)


    start_epoch = 1
    # args.max_accuracy = 0.0
    args.start_epoch = start_epoch
    logger.info("Start training")
    model.zero_grad()
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        # train
        train_loss, train_acc = train_one_epoch_local_data(train_loader, val_loader, model, loss_function, optimizer, scheduler, epoch, logger, args, tb_writer)
        save_checkpoint(epoch, model, optimizer, args.max_accuracy, args, logger, save_name='Latest'+'-epoch'+str(epoch))
        
        # validate
        logger.info(f"**********Latest val***********")
        val_loss, val_acc = validate(val_loader, model, loss_function, epoch, logger, args, tb_writer)
        # 保存最好效果
        if val_acc > args.max_accuracy:
            args.max_accuracy = val_acc
            logger.info(f'Max accuracy: {args.max_accuracy:.4f}')
            save_checkpoint(epoch, model, optimizer, args.max_accuracy, args, logger, save_name='Best')
        logger.info('Exp path: %s' % args.path_log)

    # 总时间
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch_local_data(train_loader, val_loader, model, loss_function, optimizer, scheduler, epoch, logger, args, tb_writer):
    model.train()

    num_steps = len(train_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    start = time.time()
    end = time.time()
    for iter, (images, target) in enumerate(train_loader):
        images = images.to(args.device)
        target = target.to(args.device)
        optimizer.zero_grad()

        output = model(images)  # return logits and attn, only need logits
        loss = loss_function(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        loss.backward()
        # 解决梯度爆炸！！！
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        scheduler.step()  # 更新lr
        optimizer.step()

        # 储存batch_time和loss
        batch_time.update(time.time() - end)  # 记录每次迭代batch所需时间
        end = time.time()
        loss_meter.update(loss.item(), output.size(0))  # output.size(0)
        acc1_meter.update(acc1.item(), output.size(0))
        acc5_meter.update(acc5.item(), output.size(0))
        tb_writer.add_scalar('train_loss', loss.item(), (epoch-1) * num_steps + iter)
        tb_writer.add_scalar('train_acc', acc1.item(), (epoch-1) * num_steps + iter)
        tb_writer.add_scalar('train_lr', scheduler.get_lr()[0], (epoch-1) * num_steps + iter)
        # log输出训练参数
        if iter % 50 == 0:
            etas = batch_time.avg * (num_steps - iter)
            logger.info(
                f'Train: [{epoch}/{args.epochs}][{iter}/{num_steps}]\t'
                # f'Eta {datetime.timedelta(seconds=int(etas))}\t'
                f'Time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})')

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    logger.info(f"loss: {loss_meter.avg:.4f}, acc1: {acc1_meter.avg:.4f}, acc5: {acc5_meter.avg:.4f}")
    return loss_meter.avg, acc1_meter.avg

@torch.no_grad()
def validate(val_loader, model, loss_function, epoch, logger, args, tb_writer=None):
    # switch to evaluate mode
    logger.info('eval epoch {}'.format(epoch))
    model.eval()

    num_steps = len(val_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    end = time.time()
    for iter, (images, target) in enumerate(val_loader):
        images = images.to(args.device)
        target = target.to(args.device)

        output = model(images)  # return logits and attn, only need logits
        loss = loss_function(output, target)

        # 更新记录
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        loss_meter.update(loss.item(), output.size(0))
        acc1_meter.update(acc1.item(), output.size(0))
        acc5_meter.update(acc5.item(), output.size(0))
        tb_writer.add_scalar('val_loss', loss.item(), (epoch-1) * num_steps + iter)
        tb_writer.add_scalar('val_acc', acc1.item(), (epoch-1) * num_steps + iter)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # log输出测试参数
        if iter % 50 == 0:
            logger.info(
                f'Test: [{iter}/{len(val_loader)}]\t'
                f'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})')
    logger.info(f'Eval Avg: acc@1 {acc1_meter.avg:.3f} acc@5 {acc5_meter.avg:.3f}')
    return loss_meter.avg, acc1_meter.avg


if __name__ == "__main__":
    args = parse_option()
    main(args)