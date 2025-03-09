# TMQ: Quantization-Aware Parsimonious Neural Networks

## Overview

TMQ is a Python and CUDA/C++-based framework that enables quantization-aware training of neural networks while maintaining a seamless integration into existing PyTorch workflows. With TMQ, you can:

- **Easily replace standard PyTorch layers** with quantization-aware counterparts.
- **Reduce memory footprint and computational cost** by using ternary weight quantization (<2 bits per weight on average).
- **Perform inference** directly from the compact representation with optimized CUDA kernels.
- **Improve deployment efficiency** with compressed ternary models, ideal for edge and embedded systems.

TMQ implements a differentiable transfer function to gradually push model weights toward ternary values during training, improving efficiency of neural networks.

You can give it a quick tryout on a [Google Colab](https://colab.research.google.com/drive/191iOGlUAb7QgFkuv0eGTiRc6NyrjqZp2?usp=sharing).


## Does it work ?

Here are some quick results (to be expanded) on some network/dataset combination:

| Network    | Dataset     | Top-1 acc | Top-3 acc | Weights  |                                              Size on disk |
|------------|-------------|-----------|-----------|----------|----------------------------------------------------------:|
| ResNet-18  | CalTech-101 | 80%       | 89%       | ternary  |    [2.3M](samples/data/resnet18_caltech101_compressed.pt) |
| ResNet-18  | CalTech-101 | 73%       | 84%       | fp32     |      [44M](samples/data/resnet18_caltech101_fp32_full.pt) |


## How It Works

TMQ replaces conventional weight layers with quantization-aware layers that leverage a soft-staircase transfer function to gradually map weights towards
quantized (currently ternary) values. This approach is not new and similar ideas have for example been published [here](https://arxiv.org/abs/1908.05033).
Compared to the work cited above, TMQ uses a different transfer function - based on piecewise polynomials for improved efficiency on CUDA devices (no usage of SFUs) - and a heuristic
schedule for quantization hardness/softness as opposed to a learned schedule.

## Installation

TMQ requires PyTorch and a CUDA-compatible GPU. To install TMQ, you can just fetch it from PyPi, using the following command:

```
pip install --no-build-isolation tmq
```

or just clone this repository and build it manually:

```
pip install -r requirements.txt
python setup.py install
```

## Quick Start Guide

### Convert a Standard PyTorch Model

To integrate TMQ into an existing PyTorch model, simply replace standard layers with TMQ's quantization-aware versions using `quantize_model`:

```python
import torch
import torch.nn as nn
import tmq
from tmq import QuantizationControl, QuantizationScheduleBuilder, quantize_model

# Define a simple PyTorch model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 10)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(10)
    
    def forward(self, x):
        x = self.bn1(self.fc1(x))
        x = torch.relu(x)
        return self.bn2(self.fc2(x))


# Initialize the model and TMQ controller, also create a schedule for training
model = MyModel()
schedule = QuantizationScheduleBuilder.default_schedule(max_epochs)
ctrl = QuantizationControl(schedule, device="cuda")

# Replace standard layers with quantization-aware layers
quantize_model(model, ctrl)
```

### Training with TMQ

Once the model is converted, training proceeds as usual. TMQ will progressively quantize the weights during training.

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

def train_step(batch, epoch):
    ctrl.step(epoch)
    optimizer.zero_grad()
    output = model(batch['input'])
    # add an additional quantization loss to the loss function
    loss = criterion(output, batch['label']) + ctrl.quantization_loss()
    loss.backward()
    optimizer.step()
    # clamp weights to valid range
    ctrl.clamp()
    return loss.item()
```

### Save and Load a Compact Model

After training, TMQ allows exporting the quantized weights in a compact format:

```python
# Save the quantized model
state_dict = tmq.compressed_state_dict(model)
torch.save(state_dict, "tmq_model.pth")
```

## Samples
We provide a few samples in the [samples](samples) folder, more samples to come soon.


## Understanding BatchNorm's Role in TMQ

Batch normalization (BatchNorm) is crucial when using quantized weights because it helps maintain activation variances within a reasonable range. This ensures that gradient updates 
remain stable and prevents exploding or vanishing activations. If replacing layers with TMQ quantized versions, it is **strongly recommended** to keep BatchNorm layers in your model
which are directly following the quantized layers because all accumulations are done with the weights being integer values and
a large amount of channels easily outgrows usual assumptions about the variance of activations passed between layers.

## Future Work
Quantized-weight networks are an interesting topic to work on and a few things come to mind on how to leverage them for
multiple different aspects:

### Hardware
Ternary quantization bears an important advantage in terms of compute. Not only are the weights smaller, but matrix-multiplication as backbone of
every network (linear layers, convolution layers, etc) turns into basic addition and subtraction operations. In chip-design - or when just considering building a pipeline on 
an FPGA - this means that significantly less on-chip resources like FP-multipliers are required, which leads to:
 - Fitting more units on a chip or use smaller chips and/or older processes to fit a decent amount of compute
 - Power efficiency, good for mobile/edge operation
 - Easier to build custom FPGA pipelines on cheaper FPGAs

### Parsimony
Many neural networks have too much capacity for the tasks they are supposed to handle. It is quite common to use 16-bit FP values for weights and
the trend on those goes downwards, many local LLMs run with 4-bit quantized weights and still exhibit great task-performance. With ternary weights,
the hyperparameter to optimize reduces to just the number of weights, the bit-width of the weights become less relevant. Fine-tuning networks
then boils down to choose the right amount of parameters per layer without worrying if too much redundant capacity is added.

### Activation Quantization
Quantizing weights is just half the work. Adding quantized activations and trying to get rid of floating-point math as much as possible will lead
to more efficiency in terms of compute and also memory bandwidth. Experimenting with 8-bit wide or more narrow activations is not new and adding
this as an easy-to-use feature is on the todo list. This relates closely to the next point:

### Improving CUDA Kernels
TMQ has a few custom CUDA kernels which are not particularly well optimized regarding the activation data path. Due to the high variance after
accumulation, we currently have to use FP32 to pass into the following batchnorm layers. It would be better to directly integrate the batchnorm
part into the TMQ layer itself to downscale to FP16 or smaller during the accumulation process. Also, convolutions are currently done 
using im2col and especially for smaller kernels (which are most common) additional optimization could be done to circumvent the use
of im2col. 


## License

This project is licensed under the **MPL-2.0** License - see the [LICENSE](LICENSE) file for details.


## Contribute & Support

If you find TMQ useful, please give it a star on GitHub! Contributions, pull requests, and feature suggestions are always welcome.

- GitHub Issues: [Report a Bug](https://github.com/mtnwrw/tmq/issues)

