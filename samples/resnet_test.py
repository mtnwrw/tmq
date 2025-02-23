# vim: set ts=4 sw=4 :
# _*_ coding: utf-8 _*_
# Copyright (c) 2025 TMQ Authors
# SPDX-License-Identifier: MPL-2.0

__author__ = 'Martin Wawro'

"""
Example on how to compress / decompress ResNet model files and use them for inference
"""

from argparse import ArgumentParser
import sys

import torch

from torchvision import io as tvio
import torchvision.transforms as transforms
from torchvision.models import resnet18

from tmq import QuantizationControl, TMQLayer, quantize_model, compressed_state_dict, load_compressed_state_dict

def read_class_labels(filename):
    label_dict = {}
    try:
        with open(filename, 'r') as file:
            for i, line in enumerate(file):
                label_dict[i] = line.strip()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
    except Exception as e:
        print(f"Error reading file: {e}")
    return label_dict



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--device", default="cuda", help="Device to use for inference, default is cuda (GPU)")
    parser.add_argument('--dataset', type=str, default="caltech101", help="Dataset to train on, must be either caltech101 (default) or caltech256")
    parser.add_argument('--scale', default=False, action="store_true", dest="scale", help="Perform post-scaling after MM/conv (usually handled by BN, use if BN is not used ubiquitously)")
    parser.add_argument("--disablebn", default=False, action="store_true", dest="disablebn", help="By default we append a 1D batchnorm to the final FC layer, this flag disables it")
    parser.add_argument("--download", default=False, action="store_true", dest="download", help="Download sample data if not on system")
    parser.add_argument("--checkpoint", help="Checkpoint file to load the non-compact/uncompressed model from for compression")
    parser.add_argument("--model", help="Compressed model to load")
    parser.add_argument("--cmodel", help="Filename for compressed model data")
    parser.add_argument("--umodel", help="Filename for uncompressed model data (to compare sizes)")
    parser.add_argument("--nocompress", default=False, action="store_true", dest="nocompress", help="Do not compress the weights, only useful with --inference to check the original 32-bit FP")
    parser.add_argument("--inference", type=str, help="Run inference on provided sample image")
    args = parser.parse_args()

    if args.dataset == "caltech256":
        num_classes = 256
        class_labels = None
    else:
        num_classes = 101
        class_labels = read_class_labels("caltech101.txt") if args.inference is not None else None

    # ---------------------------------------------------------
    # Instantiate quantization controller, create an empty
    # ResNet model and convert the layers into quantized ones
    # ---------------------------------------------------------
    ctrl = QuantizationControl(device=args.device)
    assert ctrl, "Quantization controller is required"

    model = resnet18(num_classes=num_classes)

    quantize_model(model, ctrl)                 # Performs an in-place quantization of the (original) model
    if not args.disablebn:
        assert isinstance(model.fc, TMQLayer), "Something went wrong, FC layer is not quantized"
        model.fc.disable_bias()
        model.fc.implicit_bn()

    # ---------------------------------------------------------
    # Load standard checkpoint (uncompressed) and optionally
    # write out a compressed model
    # ---------------------------------------------------------
    if args.checkpoint:
        cp = torch.load(args.checkpoint, map_location=args.device)
        assert cp, "Could not load checkpoint"
        model.load_state_dict(cp["model"])
        if args.umodel:
            torch.save(cp["model"], args.umodel)
        if not args.nocompress:
            ctrl.quantize()
            comp_dict = compressed_state_dict(model, True)
            if args.cmodel:
                # Write compressed model to disk
                print("Writing compressed model to %s" % args.cmodel)
                torch.save(comp_dict, args.cmodel)
    elif args.model:
        sdict = torch.load(args.model)
        assert sdict, "Could not load model"
        load_compressed_state_dict(model, sdict)
    else:
        print("Error, either provide a checkpoint or a compressed model file")
        sys.exit(1)

    # ---------------------------------------------------------
    # Run inference on a test image (model is ternary at this
    # point)...
    # ---------------------------------------------------------
    if args.inference:
        assert class_labels, "Could not load class labels"
        model = model.to(args.device)
        model.eval()
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((224, 224))])
        img = tvio.read_image(args.inference)
        result = model(transform(img.float() / 255).to(args.device).unsqueeze(0))
        classprob = torch.softmax(result, dim=1).detach().cpu()
        top3_prob, top3_cls = torch.topk(classprob, 3, dim=1)
        top3_prob = top3_prob.squeeze()
        top3_cls = top3_cls.squeeze()
        for candidate in range(3):
            cls = top3_cls[candidate].item()
            print('%s (%d): %d%%' % (class_labels[cls], cls, int(top3_prob[candidate] * 100)))


