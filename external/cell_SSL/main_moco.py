#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import builtins
import json
import math
import os
import random
import shutil
import time
import warnings
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as torchvision_models
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from torchvision.models import ResNet50_Weights, ResNet18_Weights

import vits
import moco
from dataset_utils import create_dataset, compute_and_save_mean_std
from augmentations import RotationCrop, GaussianBlur, HEStainAugmentationPil

torchvision_model_names = sorted(
    name
    for name in torchvision_models.__dict__
    if name.islower()
    and not name.startswith("__")
    and callable(torchvision_models.__dict__[name])
)

model_names = [
    "vit_small",
    "vit_base",
    "vit_conv_small",
    "vit_conv_base",
] + torchvision_model_names

parser = argparse.ArgumentParser(description="MoCo ImageNet Pre-Training")
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnet50",
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: resnet50)",
)
parser.add_argument(
    "-j",
    "--workers",
    default=32,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 32)",
)
parser.add_argument(
    "--epochs", default=100, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "--augmentation",
    default="v1",
    type=str,
    choices=["v1", "v2", "image224"],
    help="augmentation used (default: v1)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=4096,
    type=int,
    metavar="N",
    help="mini-batch size (default: 4096), this is the total "
    "batch size of all GPUs on all nodes when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.6,
    type=float,
    metavar="LR",
    help="initial (base) learning rate",
    dest="lr",
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-6,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-6)",
    dest="weight_decay",
)
parser.add_argument(
    "-p",
    "--print-freq",
    default=10,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--imgnet_init",
    default=False,
    type=bool,
    help="initializes the model with imagenet weights by DOWNLOADING them",
)
parser.add_argument(
    "--pretrained",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--world-size",
    default=-1,
    type=int,
    help="number of nodes for distributed training",
)
parser.add_argument(
    "--rank", default=-1, type=int, help="node rank for distributed training"
)
parser.add_argument(
    "--dist-url",
    default="tcp://224.66.41.62:23456",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend", default="nccl", type=str, help="distributed backend"
)
parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
parser.add_argument("--tag", default=None, type=str, help="Tag for the model trained.")
parser.add_argument(
    "--multiprocessing-distributed",
    action="store_true",
    help="Use multi-processing distributed training to launch "
    "N processes per node, which has N GPUs. This is the "
    "fastest way to use PyTorch for either single node or "
    "multi node data parallel training",
)

# Data arguments
parser.add_argument("--data_path", metavar="DIR", help="path to dataset")
parser.add_argument("--output_path", metavar="DIR", help="path to trained models")

parser.add_argument(
    "--list_slides", nargs="+", help="list of slides to use for training."
)
# moco specific configs:
parser.add_argument(
    "--moco-dim", default=256, type=int, help="feature dimension (default: 256)"
)
parser.add_argument(
    "--moco-mlp-dim",
    default=4096,
    type=int,
    help="hidden dimension in MLPs (default: 4096)",
)
parser.add_argument(
    "--moco-m",
    default=0.99,
    type=float,
    help="moco momentum of updating momentum encoder (default: 0.99)",
)
parser.add_argument(
    "--moco-m-cos",
    action="store_true",
    help="gradually increase moco momentum to 1 with a " "half-cycle cosine schedule",
)
parser.add_argument(
    "--moco-t", default=1.0, type=float, help="softmax temperature (default: 1.0)"
)

# vit specific configs:
parser.add_argument(
    "--stop-grad-conv1",
    action="store_true",
    help="stop-grad after first conv, or patch embedding",
)

# other upgrades
parser.add_argument(
    "--optimizer",
    default="lars",
    type=str,
    choices=["lars", "adamw"],
    help="optimizer used (default: lars)",
)
parser.add_argument(
    "--warmup-epochs", default=10, type=int, metavar="N", help="number of warmup epochs"
)
parser.add_argument(
    "--crop-min",
    default=0.08,
    type=float,
    help="minimum scale for random cropping (default: 0.08)",
)

# for experiments
parser.add_argument(
    "--n_cell_max",
    default=1_000_000,
    type=int,
    help="number maximum of cells to use for training",
)


def main(args):
    # save all arguments as a json file
    os.makedirs(os.path.join(args.output_path, "config"), exist_ok=True)
    with open(os.path.join(args.output_path, "config", f"{args.tag}.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    # ngpus_per_node = torch.cuda.device_count()
    ngpus_per_node = 1
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not first GPU on each node
    if args.multiprocessing_distributed and (args.gpu != 0 or args.rank != 0):

        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
            dist.init_process_group(
                backend=args.dist_backend,
                init_method=args.dist_url,
                world_size=args.world_size,
                rank=args.rank,
            )
    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch.startswith("vit"):
        model = moco.builder.MoCo_ViT(
            partial(vits.__dict__[args.arch], stop_grad_conv1=args.stop_grad_conv1),
            args.moco_dim,
            args.moco_mlp_dim,
            args.moco_t,
        )
    else:
        # downloads the imagenet weights
        if args.imgnet_init:
            if args.arch == "resnet50":
                weights = ResNet50_Weights.IMAGENET1K_V2
            elif args.arch == "resnet18":
                weights = ResNet18_Weights.IMAGENET1K_V1
        model = moco.builder.MoCo_ResNet(
            partial(
                torchvision_models.__dict__[args.arch],
                zero_init_residual=True,
                weights=weights if args.imgnet_init else None,
            ),
            args.moco_dim,
            args.moco_mlp_dim,
            args.moco_t,
        )

    # infer learning rate before changing batch size
    args.lr = args.lr * args.batch_size / 256

    if not torch.cuda.is_available():
        print("using CPU, this will be slow")
    elif args.distributed:
        # apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu]
            )
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather/rank implementation in this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    print(model)  # print model after SyncBatchNorm

    if args.optimizer == "lars":
        optimizer = moco.optimizer.LARS(
            model.parameters(),
            args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
        )
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), args.lr, weight_decay=args.weight_decay
        )

    save_dir = os.path.join(args.output_path, args.tag)
    os.makedirs(save_dir, exist_ok=True)

    scaler = torch.cuda.amp.GradScaler()
    summary_writer = SummaryWriter(log_dir=save_dir) if args.rank == 0 else None

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scaler.load_state_dict(checkpoint["scaler"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # optionally load any pretrained model
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            if args.gpu is None:
                checkpoint = torch.load(args.pretrained)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.pretrained, map_location=loc)
            model.load_state_dict(checkpoint["state_dict"])

    checkpointdir = os.path.join(save_dir, "checkpoints")
    print(checkpointdir)
    os.makedirs(checkpointdir, exist_ok=True)

    if args.augmentation == "v1":
        augmentation = [
            transforms.RandomApply(
                [transforms.ColorJitter(0.6, 0.7, 0.5, 0.2)], p=0.8  # not strengthened
            ),
            transforms.RandomApply([RotationCrop(360, 48)], p=1),  # Hard coded !
            transforms.RandomResizedCrop(48, scale=(0.6, 1.0)),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.1),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.3, scale=(0.05, 0.15), ratio=(0.8, 1.2)),
        ]
    elif args.augmentation == "v2":
        augmentation = [
            transforms.RandomApply([HEStainAugmentationPil(1., 0.3)], p=1),  # Hard coded !
            transforms.RandomApply(
                [transforms.ColorJitter(0.6, 0.7, 0.5, 0.2)], p=0.8  # not strengthened
            ),
            transforms.RandomApply([RotationCrop(360, 48)], p=1),  # Hard coded !
            transforms.RandomResizedCrop(48, scale=(0.6, 1.0)),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.1),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.3, scale=(0.05, 0.15), ratio=(0.8, 1.2)),
        ]
    elif args.augmentation == "image224":
        # image 224 x 224 --> resize
        augmentation = [
            transforms.Resize(300),
            transforms.RandomApply(
                [transforms.ColorJitter(0.6, 0.7, 0.5, 0.2)], p=0.8  # not strengthened
            ),
            transforms.RandomApply([RotationCrop(360, 224)], p=1),  # Hard coded !
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.1),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.3, scale=(0.05, 0.15), ratio=(0.8, 1.2)),
        ]
    else:
        raise NotImplementedError("Augmentation not implemented")

    print(f"Using augmentation: {args.augmentation}")

    mean_std_path = os.path.join(save_dir, "moco_model_best_mean_std.json")
    with open(mean_std_path, "r") as f:
        mean_std = json.load(f)
        mean = mean_std["mean"]
        std = mean_std["std"]

    # Data loading code
    normalize = transforms.Normalize(mean=mean, std=std)
    augmentation.append(normalize)
    train_dataset = create_dataset(
        args.data_path,
        args.list_slides,
        moco.loader.TwoCropsTransform(
            transforms.Compose(augmentation), transforms.Compose(augmentation)
        ),
     )

    # Keep only n_cell_max cells
    print(f"Found {len(train_dataset)} cells for training.")
    if args.n_cell_max is not None and (len(train_dataset) > args.n_cell_max):
        indices = np.random.choice(len(train_dataset), args.n_cell_max, replace=False)
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        print(f"Using only {args.n_cell_max} cells for training.")

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # set loss to inf
    best_loss = float("inf")
    best_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        avg_loss = train(
            train_loader, model, optimizer, scaler, summary_writer, epoch, args
        )
        print(f"Epoch {epoch} - Average Loss: {avg_loss}")

        if not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank == 0
        ):  # only the first GPU saves checkpoint
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                },
                filename=os.path.join(checkpointdir, "checkpoint_%04d.pth.tar" % epoch),
            )
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_epoch = epoch

    if args.rank == 0:
        summary_writer.close()

    # print("Don't forget to uncommand after time test.")
    # Copy last checkpoints
    shutil.copyfile(
        os.path.join(checkpointdir, "checkpoint_%04d.pth.tar" % (best_epoch)),
        os.path.join(save_dir, f"moco_model_best.pth.tar"),
    )


def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        print(f"Time taken: {elapsed:.6f} seconds")
        return result

    return wrapper


@timeit
def train(train_loader, model, optimizer, scaler, summary_writer, epoch, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    learning_rates = AverageMeter("LR", ":.4e")
    losses = AverageMeter("Loss", ":.4e")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, learning_rates, losses],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()

    end = time.time()
    iters_per_epoch = len(train_loader)
    moco_m = args.moco_m
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # adjust learning rate and momentum coefficient per iteration
        lr = adjust_learning_rate(optimizer, epoch + i / iters_per_epoch, args)
        learning_rates.update(lr)
        if args.moco_m_cos:
            moco_m = adjust_moco_momentum(epoch + i / iters_per_epoch, args)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(True):
            loss = model(images[0], images[1], moco_m)

        losses.update(loss.item(), images[0].size(0))
        if args.rank == 0:
            summary_writer.add_scalar("loss", loss.item(), epoch * iters_per_epoch + i)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return losses.avg


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def adjust_learning_rate(optimizer, epoch, args):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = (
            args.lr
            * 0.5
            * (
                1.0
                + math.cos(
                    math.pi
                    * (epoch - args.warmup_epochs)
                    / (args.epochs - args.warmup_epochs)
                )
            )
        )
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def adjust_moco_momentum(epoch, args):
    """Adjust moco momentum based on current epoch"""
    m = 1.0 - 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs)) * (
        1.0 - args.moco_m
    )
    return m


if __name__ == "__main__":
    args = parser.parse_args()
    compute_and_save_mean_std(args)
    main(args)
    print("End of python script.")
