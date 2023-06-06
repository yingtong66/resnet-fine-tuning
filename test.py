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

def setup(args, logger):
    checkpoint = torch.load("%s" % args.load)['model']

    resnet50 = models.resnet50(num_classes=128)
    dim_mlp = resnet50.fc.weight.shape[1]
    print(dim_mlp)
    resnet50.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), resnet50.fc)
    
    resnet50.fc = nn.Linear(dim_mlp, 40)  # 将输出类别数设置为40
    # resnet50.fc=nn.Sequential(nn.Dropout(p=0.5),nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, 40))
    resnet50.load_state_dict(checkpoint)
    resnet50.to(args.device)
    
    logger.info("Training parameters %s", args)
    return args, resnet50


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def parse_option():
    parser = argparse.ArgumentParser('Resnet50 testing script', add_help=False)
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
    parser.add_argument("--lr", default=1e-3, type=float, help="The initial learning rate for SGD.")
    parser.add_argument("--max_accuracy", default=0.0, type=float)
    # model
    parser.add_argument("--pretrain_dir", type=str, default="./pretrain", help="Where to search for pretrained ViT models.")
    parser.add_argument('--pretrain', type=str, default="ViT-B_16.npz", help='vit_base_patch16_224_in21k.pth')
    parser.add_argument('--model_file', type=str, default='modeling')
    parser.add_argument("--load", default="", type=str, help="Load model from a .pth file")
    # 是否冻结权重

    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
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
    train_loader, eval_loader, test_loader = get_loader(args)

    # print args
    for param in sorted(vars(args).keys()):  # 遍历args的属性对象
        logger.info('--{0} {1}'.format(param, vars(args)[param]))

    # test
    logger.info(f"**********Test***********")
    val_loss, val_acc = test(test_loader, model, logger, args)

@torch.no_grad()
def test(test_loader, model, logger, args):
    model.eval()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    for iter, (images, target) in enumerate(test_loader):
        images = images.to(args.device)
        target = target.to(args.device)

        output = model(images)  # return logits and attn, only need logits
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        acc1_meter.update(acc1.item(), output.size(0))
        acc5_meter.update(acc5.item(), output.size(0))
        # log输出测试参数
    logger.info(f'Test Avg: acc@1 {acc1_meter.avg:.3f} acc@5 {acc5_meter.avg:.3f} finished')
    return acc1_meter.avg, acc5_meter.avg


if __name__ == "__main__":
    args = parse_option()
    main(args)