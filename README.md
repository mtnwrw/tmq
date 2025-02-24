# TMQ: Quantization-Aware Parsimonous Neural Networks

## Overview

TMQ is a Python and CUDA/C++-based framework that enables quantization-aware training of neural networks while maintaining a seamless integration into existing PyTorch workflows. With TMQ, you can:

- **Easily replace standard PyTorch layers** with quantization-aware counterparts.
- **Reduce memory footprint and computational cost** by using ternary weight quantization (<2 bits per weight on average).
- **Perform inference** directly from the compact representation with optimized CUDA kernels.
- **Improve deployment efficiency** with compressed ternary models, ideal for edge and embedded systems.

TMQ implements a differentiable transfer function to gradually push model weights toward ternary values during training, improving efficiency of neural networks.

## How It Works

TMQ replaces conventional weight layers with quantization-aware layers that leverage a soft-staircase transfer function to gradually map weights towards
quantized (currently ternary) values. This approach is not new and approaches like [this](https://arxiv.org/abs/1908.05033) are nearly identical to the approach taken here.
Compared to the work cited above, TMQ uses a different transfer function based on piecewise polynomials for improved efficiency on CUDA devices (no usage of SFUs) and a heuristic
schedule for quantization hardness/softness as opposed to a learned schedule.

The benefits of ternary quantization are:
- **Memory Efficiency:** Less than 2 bits per weight, reducing storage and bandwidth requirements.
- **Multiplication-Free Inference:** Enables efficient matrix multiplications using only additions and subtractions.
- **Improved Hardware Deployment:** Ternary-weight networks are well-suited for FPGA and ASIC implementations.

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
which are directly following the quantized layers.

## License

This project is licensed under the **MPL-2.0** License - see the [LICENSE](LICENSE) file for details.


## Contribute & Support

If you find TMQ useful, please give it a star on GitHub! Contributions, pull requests, and feature suggestions are always welcome.

- GitHub Issues: [Report a Bug](https://github.com/mtnwrw/tmq/issues)

