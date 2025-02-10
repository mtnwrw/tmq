# vim: set ts=4 sw=4 :
# _*_ coding: utf-8 _*_
# Copyright (c) 2025 TMQ Authors
# SPDX-License-Identifier: MPL-2.0

__author__ = 'Martin Wawro'

"""
Sample how to use TMQ on a vanilla ResNet-18 image classifier
"""

from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import sys
import os
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Dataset, Subset

from torchvision import transforms, datasets
from torchvision.models import resnet18
from loaders.caltech import Caltech101, Caltech256

from tmq import QuantizationControl, QuantizationScheduleBuilder, SoftStep, TMQLayer, quantize_model, compressed_state_dict



def init_wandb(project_name, run_name, config, key):
    """
    Initializes weights & biases connection for logging (if enabled)
    """
    wandb.login(key=key)
    log_config = config
    del log_config.wbkey
    wandb.init(project=project_name, name=run_name, config=log_config)
    return True


def log_epoch(model, optim, scheduler, train_loss, val_loss, top1, top3, val_top1, val_top3, ctrl, qloss, epoch):
    """

    Cave: this function is quite hacky and assumes a certain naming of the modules inside the network it logs
          (as it uploads histograms of weight distributions of a couple of layers). If for any reason the names of
          those layers change in the torchvision model, this function will fail.

    :param model: Model that undergoes optimization
    :param optim: Optimizer used in the training
    :param scheduler: Optional scheduler used in the training
    :param train_loss: Current (total) training loss in this epoch
    :param val_loss: Current validation loss in this epoch
    :param top1: Top-1 accuracy for the training dataset
    :param top3: Top-3 accuracy for the training dataset
    :param val_top1:  Top-1 accuracy for the validation dataset
    :param val_top3: Top-3 accuracy for the validation dataset
    :param ctrl: Optional quantization control instance used for quantized training
    :param qloss: Optional quantization loss
    :param epoch: Current epoch
    """
    bins = 63
    conv1raw = model.conv1.weight.data.detach()
    layer1c2raw = model.layer1[0].conv2.weight.data.detach()
    layer3c1raw = model.layer3[0].conv1.weight.data.detach()
    layer4c1raw = model.layer4[0].conv1.weight.data.detach()
    layer4c2raw = model.layer4[0].conv2.weight.data.detach()
    fcraw = model.fc.weight.data.detach()
    ldict = {
        "train/digamma": ctrl.digamma if ctrl is not None else 0,
        "train/qpenalty": ctrl.qpenalty if ctrl is not None else 0
    }
    if train_loss is not None and epoch > 0:
        ldict["train/loss"] = train_loss
        ldict["train/top1"] = top1
        ldict["train/top3"] = top3
        ldict["val/loss"] = val_loss
        ldict["val/top1"] = val_top1
        ldict["val/top3"] = val_top3
        ldict["train/lr"] = scheduler.get_last_lr()[0] if scheduler is not None else optim.param_groups[0]['lr']
        if qloss is not None:
            ldict["train/qloss"] = qloss

    ldict["conv1raw"] = wandb.Histogram(np_histogram=np.histogram(conv1raw.cpu().numpy(), bins=bins))
    ldict["l1conv2raw"] = wandb.Histogram(np_histogram=np.histogram(layer1c2raw.cpu().numpy(), bins=bins))
    ldict["l3conv1raw"] = wandb.Histogram(np_histogram=np.histogram(layer3c1raw.cpu().numpy(), bins=bins))
    ldict["l4conv1raw"] = wandb.Histogram(np_histogram=np.histogram(layer4c1raw.cpu().numpy(), bins=bins))
    ldict["l4conv2raw"] = wandb.Histogram(np_histogram=np.histogram(layer4c2raw.cpu().numpy(), bins=bins))
    ldict["fcraw"] = wandb.Histogram(np_histogram=np.histogram(fcraw.cpu().numpy(), bins=bins))
    if isinstance(model.conv1, TMQLayer):
        ldict["conv1step"] = wandb.Histogram(np_histogram=np.histogram(SoftStep.map(conv1raw, ctrl).cpu().numpy(), bins=bins))
        ldict["l1conv2step"] = wandb.Histogram(np_histogram=np.histogram(SoftStep.map(layer1c2raw, ctrl).cpu().numpy(), bins=bins))
        ldict["l3conv1step"] = wandb.Histogram(np_histogram=np.histogram(SoftStep.map(layer3c1raw, ctrl).cpu().numpy(), bins=bins))
        ldict["l4conv1step"] = wandb.Histogram(np_histogram=np.histogram(SoftStep.map(layer4c1raw, ctrl).cpu().numpy(), bins=bins))
        ldict["l4conv2step"] = wandb.Histogram(np_histogram=np.histogram(SoftStep.map(layer4c2raw, ctrl).cpu().numpy(), bins=bins))
        ldict["fcstep"] = wandb.Histogram(np_histogram=np.histogram(SoftStep.map(fcraw, ctrl).cpu().numpy(), bins=bins))

    wandb.log(ldict)


def train_single_epoch(model, ctrl, optim, loader, epoch, device, class_weights=None):
    """
    Run all batches in a single epoch for training

    :param model: Model to train
    :param ctrl: Optional quantization controller
    :param optim: Optimizer used in training
    :param loader: Dataloader
    :param epoch: Current epoch
    :param device: Device to run on
    :param class_weights:

    :return: Tuple of (averaged) losses incurred during training batch
    """
    model.train()
    if ctrl is not None:
        ctrl.clamp()
    avgloss, avgqloss, top1_acc, top3_acc = 0, 0, 0, 0
    for batch_idx, item in enumerate(tqdm(loader)):
        optim.zero_grad()
        pred = model(item[0].to(device))
        loss = F.cross_entropy(pred, item[1].to(device), weight=class_weights)
        if ctrl is not None:
            qloss = ctrl.quantization_loss()
            loss += qloss
        else:
            qloss = None

        loss.backward()
        optim.step()
        if ctrl is not None:
            ctrl.clamp()

        p = pred.detach()
        _, top1_inds = torch.topk(p, 1)
        _, top3_inds = torch.topk(p, 3)
        top1_inds = top1_inds.cpu()
        top3_inds = top3_inds.cpu()
        gt = item[1].detach().cpu().view(-1, 1)
        top1_hits = top1_inds == gt
        top3_hits = top3_inds == gt
        top3_hits = top3_hits.sum(dim=1).float()
        top1_acc += top1_hits.float().mean().item()
        top3_acc += top3_hits.mean().item()

        avgloss += loss.item()
        avgqloss += qloss.item() if qloss is not None else 0
        if batch_idx % 50 == 0:
            tqdm.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tAvgLoss: {:.6f}.'.format(epoch, batch_idx, len(loader),
                                                                              100. * batch_idx / len(loader), avgloss / (batch_idx+1)))
    return avgloss / len(loader), avgqloss / len(loader), top1_acc / len(loader), top3_acc / len(loader)


def validate_single_epoch(model, loader, epoch, device, class_weights=None):
    """
    Run a batch of tests
    """
    model.eval()
    avgloss, top1_acc, top3_acc = 0, 0, 0
    for batch_idx, item in enumerate(tqdm(loader)):
        pred = model(item[0].to(device))
        loss = F.cross_entropy(pred, item[1].to(device), weight=class_weights)
        avgloss += loss.item()
        p = pred.detach()
        _, top1_inds = torch.topk(p, 1)
        _, top3_inds = torch.topk(p, 3)
        top1_inds = top1_inds.cpu()
        top3_inds = top3_inds.cpu()
        gt = item[1].detach().cpu().view(-1,1)
        top1_hits = top1_inds == gt
        top3_hits = top3_inds == gt
        top3_hits = top3_hits.sum(dim=1).float()
        top1_acc += top1_hits.float().mean().item()
        top3_acc += top3_hits.mean().item()
        if batch_idx % 25 == 0:
            tqdm.write('Test Epoch: {} [{}/{} ({:.0f}%)]\tAvgLoss: {:.6f}.'.format(epoch, batch_idx, len(loader),
                                                                              100. * batch_idx / len(loader), avgloss / (batch_idx+1)))
    return avgloss / len(loader), top1_acc / len(loader), top3_acc / len(loader)



def training_loop(model, optim, scheduler, loaders, start_epoch, max_epoch, qctrl, run, enable_wandb, device, class_weights=None):
    """
    Training loop that runs a complete batch of training data followed by a complete batch of test data



    :param model: Model to be trained
    :param optim: Optimizer to be used for training
    :param scheduler: Optional learning rate scheduler (None if no LR scheduling is desired)
    :param loaders: Tuple of data loaders (train, validation) for loading the training/test batches
    :param start_epoch: Epoch to start at
    :param max_epoch: Maximum epoch to work up to (exclusively)
    :param qctrl: Quantization control instance
    :param run: Name of the run (for logging purposes)
    :param enable_wandb: If set to True, will enable logging to W&B
    :param device: Compute device to run on
    :param class_weights:
    """
    bestloss = None

    # Log out the initial state of the model
    if enable_wandb:
        log_epoch(model, optim, scheduler, None, None, None, None, None, None, qctrl, None, 0)

    # Epoch loop
    for epoch in range(start_epoch, max_epoch):

        # Step the quantization controller
        if qctrl is not None:
            qctrl.step(epoch)

        # Run training batches
        trainloss, qloss, ttop1, ttop3 = train_single_epoch(model, qctrl, optim, loaders[0], epoch, device, class_weights)

        # Run validation batches
        loss, top1, top3 = validate_single_epoch(model, loaders[1], epoch, device, class_weights)

        # Log epoch results to W&B if enabled
        if enable_wandb:
            log_epoch(model, optim, scheduler, trainloss, loss, ttop1, ttop3, top1, top3, qctrl, qloss, epoch)

        # Though it only partially makes sense, we do keep track of a "best" model. Normally we would use
        # the state of the last epoch since this is quantized to the maximum extent
        if bestloss is None or loss < bestloss:
            torch.save({'optimizer': optim.state_dict(), 'model': model.state_dict(), 'epoch': epoch},
                       f"checkpoints/best_{run}.pt")
            bestloss = loss

        # Store the epoch checkpoint. In regard to the desired outcome (a fully quantized model), those checkpoints
        # are the ones to be used over the checkpoints with the best test loss/accuracy
        torch.save({'optimizer':optim.state_dict(), 'model': model.state_dict(), 'epoch': epoch}, f"checkpoints/checkpoint_{run}.pt")

        # If we use a learning rate scheduler, step it
        if scheduler is not None:
            scheduler.step()

    # After training is done, we force the model to be fully quantized and run another epoch (this does not run on a ternary
    # representation but is numerically equivalent to it)
    if qctrl is not None:
        qctrl.quantize()
        if enable_wandb:
            train_loss, ttop1, ttop3 = validate_single_epoch(model, loaders[0], max_epoch, device)  # we fake this for the logger
            val_loss, top1, top3 = validate_single_epoch(model, loaders[1], max_epoch, device)
            log_epoch(model, optim, scheduler, train_loss, val_loss, ttop1, ttop3, top1, top3, qctrl, None, max_epoch)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--device", default="cuda", help="Device to train on, default is cuda (GPU)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and testing (defaults to 64), adjust according to your GPU memory")
    parser.add_argument("--resume", help="Start from a checkpoint file (resume training) - currently defunct")
    parser.add_argument('--float', default=False, action="store_true", dest="float", help="Do not quantize and perform computations in 32-bit FP for reference purposes")
    parser.add_argument('--dataset', type=str, default="caltech101", help="Dataset to train on, must be either caltech101 (default) or caltech256")
    parser.add_argument('--scale', default=False, action="store_true", dest="scale", help="Perform post-scaling after MM/conv (usually handled by BN, use if BN is not used ubiquitously)")
    parser.add_argument('--wandb', type=str, default="", help="Name of W&B project (simultaneously enable logging via W&B)")
    parser.add_argument('--run', type=str, help="Name of run for W&B logging (only useful if wandb is set)")
    parser.add_argument("--max_epochs", type=int, default=150, help="Maximum number of epochs to train for (defaults to 150)")
    parser.add_argument("--lr", type=float, default=0.1, help="(Initial) learning rate, default is 0.1, currently the learning rate is kept static")
    parser.add_argument("--datadir", type=str, required=True, help="Root directory to load training/test data from")
    parser.add_argument('--qpenbase', type=float, default=0.0005, help="(No)Quantization penalty base value, defaults to 0.0005")
    parser.add_argument('--qpen', type=float, default=0.0005, help="Quantization penalty target value, defaults to 0.0005")
    parser.add_argument('--wbkey', type=str, help="Authentication key for W&B")
    parser.add_argument('--digamma', type=float, default=6, help="Final digamma (stepiness) value for in-training quantization, should not exceed 20, sensible values are around 6-10")
    parser.add_argument("--disablebn", default=False, action="store_true", dest="disablebn", help="By default we append a 1D batchnorm to the final FC layer, this flag disabled it")
    parser.add_argument("--download", default=False, action="store_true", dest="download", help="Download sample data if not on system")
    args = parser.parse_args()

    # ---------------------------------------------------------
    # Yup...
    # ---------------------------------------------------------
    torch.manual_seed(42)

    # ---------------------------------------------------------
    # Misc initializations
    # ---------------------------------------------------------
    if args.run is not None and len(args.run) > 0:
        run = args.run
    else:
        run = datetime.now().strftime('%Y%m%d-%H-%M')

    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")

    # ---------------------------------------------------------
    # Initialize Weights & Biases for logging (if enabled)
    # ---------------------------------------------------------
    if args.wandb is not None and len(args.wandb) > 0:
        try:
            import wandb
            use_wandb = init_wandb(args.wandb, run, args, args.wbkey)
        except Exception:
            use_wandb = False
    else:
        use_wandb = False

    # ---------------------------------------------------------
    # Establish datasets...
    # ---------------------------------------------------------
    transforms = transforms.Compose([transforms.Resize([224, 224]), transforms.Lambda(lambda x: x.convert('RGB')), transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    if args.dataset == "caltech256":
        print("Using CalTech-256 dataset")
        rootdata = Caltech256(args.datadir, args.download)
    else:
        print("Using CalTech-101 dataset")
        rootdata = Caltech101(args.datadir, args.download)
    num_classes = rootdata.num_classes()

    if rootdata is None or len(rootdata) == 0:
        print("No data found in %s" % args.datadir)
        sys.exit(1)

    train, val = rootdata.train_test_split(0.8, False)
    train.transform = transforms
    val.transform = transforms
    print("%d items in complete dataset" % len(rootdata))
    print("%d items in training set, %d items in validation set" % (len(train), len(val)))


    # ---------------------------------------------------------
    # Create and quantize model, optimizer and optional learning
    # rate scheduler, load state dict if in resume mode...
    # ---------------------------------------------------------
    schedule = QuantizationScheduleBuilder. default_schedule(args.max_epochs)
    ctrl = QuantizationControl(schedule, digamma_range=(0.1, args.digamma), qpenalty_range=(args.qpenbase, args.qpen), post_scale=args.scale, device=args.device) if not args.float else None

    model = resnet18(num_classes=num_classes)

    if ctrl is not None:
        quantize_model(model, ctrl)                 # Performs an in-place quantization of the model
        if not args.disablebn:
            assert isinstance(model.fc, TMQLayer), "Something went wrong, FC layer is not quantized"
            model.fc.disable_bias()
            model.fc.implicit_bn()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.float:
        # Use a learning rate scheduler for 32-bit FP reference computations
        lrscheduler = ExponentialLR(optimizer, gamma=0.99)
    else:
        lrscheduler = None

    start_epoch = 1
    if args.resume is not None:
        if len(args.resume) > 0:
            cp = torch.load(args.resume, map_location=args.device)
            assert cp
            start_epoch = cp["epoch"]
            model.load_state_dict(cp["model"])
            optimizer.load_state_dict(cp["optimizer"])
        else:
            print("Checkpoint required")
            sys.exit(1)

    model = model.to(args.device)

    # ---------------------------------------------------------
    # Create data loaders
    # ---------------------------------------------------------
    train_loader = DataLoader(train, batch_size=args.batch_size, drop_last=True, num_workers=4, shuffle=True)
    val_loader = DataLoader(val, batch_size=args.batch_size, drop_last=True, num_workers=4, shuffle=True)

    # ---------------------------------------------------------
    # Run training loop
    # ---------------------------------------------------------

    training_loop(model, optimizer, lrscheduler, (train_loader, val_loader), start_epoch, args.max_epochs,
                  ctrl, run, use_wandb, device=args.device, class_weights=None)


