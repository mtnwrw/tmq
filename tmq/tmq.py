# vim: set ts=4 sw=4 :
# _*_ coding: utf-8 _*_
# Copyright (c) 2025 TMQ Authors
# SPDX-License-Identifier: MPL-2.0

__author__ = 'Martin Wawro'

from typing import Mapping, Any

"""
This file contains drop-in replacement classes for various PyTorch operators that implement (soft-)quantization.
Currently supported are:
  - Conv2d
  - ConvTranspose2d
  - Linear
  
This file also features a control class that keeps track of all quantization-aware instances as well as providing
global parameters to the quantization operations in order to control the training process.  
"""

import torch
from torch import nn
from torch.autograd.function import Function, FunctionCtx
import fnmatch
from enum import Enum

# Import CUDA helpers if compiled
try:
    from .tmq_cuda import fwd_softstep, bwd_softstep, ternary_mmm_cuda, ternary_mvm_cuda, \
                          ternary_dwconv_cuda,  ternary_dwconvtrans_cuda, \
                          fwd_softstep_derivative, bwd_softstep_derivative
    CUDA_EXT_AVAILABLE = True
except ImportError as err:
    print("WARNING: Cannot load CUDA extension, please double-check your installation")
    print(f"Error from import: {err}")
    CUDA_EXT_AVAILABLE = False

# Import native code helpers (for compression) if compiled
try:
    from .tmq_native import compactify_ternary, compress_ternary, decompress_ternary, expand_ternary
    NATIVE_EXT_AVAILABLE = True
except ImportError as err:
    print("WARNING: Cannot load native extension, please double-check your installation")
    print(f"Error from import: {err}")
    # NOTE (mw) when native helpers are not available, we cannot compress or decompress the data
    NATIVE_EXT_AVAILABLE = False


class QuantMode(Enum):
    """
    Enumerator for quantization modes

    As of now this code-base is only tested with ternary quantization. It is not clear if the other modes will
    ever be implemented.
    """
    NONE = 0                            # No quantization
    TERNARY = 1                         # Ternary quantization {-1,0,1}
    TWO_BIT = 2                         # 2-bit quantization {-2,..,1}
    THREE_BIT = 3                       # 3-bit quantization {-4,...3}
    FOUR_BIT = 4                        # 4-bit quantization {-8,...,7}
    FIVE_BIT = 5                        # 5-bit quantization {-16,...,15}
    SIX_BIT = 6                         # 6-bit quantization {-32,...,31}
    EIGHT_BIT = 8                       # 8-bit quantization {-128,...,127}

class TensorType(Enum):
    """
    Enumerates the data type of the (stored) weight tensor for TMQ layers
    """
    FLOAT_32BIT = 0                     # Non-compact/non-compressed 32-bit float (subject to quantization)
    FLOAT_16BIT = 1                     # Non-compact/non-compressed 16-bit float (subject to quantization)
    COMPACT_TERNARY_32BIT = 2           # Compact/non-compressed 2-bit ternary bit-stuffed in a 32-bit word (16 elements per word)
    COMPRESSED_TERNARY_32BIT = 3        # Compressed 2-bit ternary bit-stuffed in a 32-bit word (16 elements per word)


class QuantizationScheduleBuilder:
    """
    Simple builder for quantization schedules, rather incomplete as of now and requires more work
    """
    @staticmethod
    def default_schedule(max_epochs : int):
        assert max_epochs >= 50, "Need at least 50 epochs for a default schedule"
        return int(max_epochs * 0.3), int(max_epochs * 0.8), int(max_epochs * 0.875), int(max_epochs * 0.93), max_epochs


class QuantizationControl:
    """
    Main controller instance for quantization-aware training

    This serves as controlling unit for all quantization-aware layers in a model. All instances of TMQLayer are
    registered in the controller (and also have stored associations to the controller themselves) in order to read
    parameters and invoke callbacks in different circumstances. Even for inference, a controller instance is required
    to orchestrate things.

    This class was initially born as a hack, but it proved itself useful, so we keep it for now.



    The epoch_knots parameter is used for modulating the digamma schedule as well as the quantization loss schedule
    during training. In particular, it provides the epochs of 5 control points during training:
      - index 0: Number of epochs to reach a digamma value of 3 (50% of the value required for a zero gradient at integer positions)
      - index 1: Number of epochs to reach the digamma max value and start modulating the quantization loss function
      - index 2: Number of epochs to reach the maximum quantization loss penalty
      - index 3: Number of epochs to stop cycling the digamma value
      - index 4: Number of total epochs for training
    """

    def __init__(self, epoch_knots=None, digamma_range=(0.1, 6), qpenalty_range=None, quant_mode=QuantMode.TERNARY, post_scale=False, init_range=None, device="cpu"):
        # TODO (mw) check if we can get rid of the device
        assert epoch_knots is None or len(epoch_knots) == 5
        assert len(digamma_range) == 2, "Need to supply a start/end value for the digamma parameter of the softstep function"
        assert digamma_range[0] >= 0.05, "Initial digamma value must be >= 0.05"
        assert digamma_range[0] <= 1, "Initial digamma value must be <= 1"
        assert digamma_range[1] <= 20.0, "Final digamma value must be <= 20"
        assert digamma_range[1] >= 3, "Final digamma value must be >= 3"
        assert digamma_range[0] <= digamma_range[1], "Please supply a valid range for digamma"

        self._warmup_epochs = 7                                # TODO (mw) make this configurable ?

        if epoch_knots is not None:
            assert epoch_knots[0] >= self._warmup_epochs+5, "At last %d epochs required to reach halfway point" % self._warmup_epochs+5
            self.max_epochs = epoch_knots[4]
        else:
            self.max_epochs = 0

        # Misc member variable initializations
        self.quantization = quant_mode                          # Quantization mode to use
        self.epoch_knots = epoch_knots                          # Control vector for the training schedule
        self.post_scale = post_scale                            # Apply fixed post-scaling on each TMQLayer instance (always prefer batchnorm before doing this)
        self.digamma_start = max(0.05, digamma_range[0])        # Starting value for quantization-softness (avoid 0 here, 0.05 is basically no quantization)
        self.digamma_end = max(3, digamma_range[1])             # Final value for quantization softness (never go below 3 which is still quite soft)
        self.digamma = self.digamma_start                       # Current value for the schedule that provides the quantization softness (or hardness, depending on what perspective)
        self.qpenalty = qpenalty_range[0] if qpenalty_range is not None else 0
        self.qpenalty_sharpness = 5.0                           # TODO (mw) make this configurable ?
        self.inference_mode = self.max_epochs == 0              # Switch to inference mode if no schedule is done, otherwise assume we are running in training
        self.device = device                                    # Device that the TMQLayer instances are running on

        self._registry = {}                                     # Registry for _all_ TMQLayer instances in the model (must be complete)
        self._last_cycle = -1
        self._digamma_cycle_start = 0
        self._digamma_cycle_end = 0
        self._cycle_len = 15                                    # HACK (mw) this is based on only a few experiments

        # Set clamping values and parameter initialization range based on number of quantization leve
        if quant_mode == QuantMode.TERNARY:
            self.clamp_values = (-1., 1.)
        else:
            bits = int(quant_mode.value)
            self.clamp_values = (-2**bits, (2**bits)-1)
        self.init_range = init_range if init_range is not None else self.clamp_values

        # Set the penalty parameter range for the quantization loss
        if qpenalty_range is not None:
            assert len(qpenalty_range) == 2, "Quantization penalty needs to supply a range"
            self.qpenalty_range = qpenalty_range
        else:
            self.qpenalty_range = (0, 0)



    def step(self, epoch):
        """
        Quantization schedule stepping/epoch function which adjusts quantization parameters per epoch

        Use this function before every epoch of training to advance the quantization parameters according to the provided
        schedule.
        """

        # ---------------------------------------------------
        # Some internal helpers functions, skip down 20 lines
        # to look at the actual implementation
        # ---------------------------------------------------
        def _lerp(alpha0, alpha1, alpha, source, target):
            weight = min(1, max(0, (alpha-alpha0) / (alpha1-alpha0)))
            return weight * target + (1-weight) * source

        def _knot_interval(tgt_epoch):
            k = 0 if tgt_epoch < self.epoch_knots[0] else 1
            k = 2 if k == 1 and tgt_epoch >= self.epoch_knots[1] else k
            k = 3 if k == 2 and tgt_epoch >= self.epoch_knots[2] else k
            k = 4 if tgt_epoch >= self.epoch_knots[3] else k
            return k

        def _digamma(knot, epoch):
            if knot >= 2:
                return self.digamma_end
            src = self.digamma_start if knot == 0 else 3.0 if knot == 1 else self.digamma_end
            tgt = 3.0 if knot == 0 else self.digamma_end
            e0 = self._warmup_epochs if knot == 0 else self.epoch_knots[knot - 1]
            e1 = self.epoch_knots[knot]
            return _lerp(e0, e1, epoch, src, tgt)

        # -------------------------------------------------
        # If we have no schedule, we cannot step
        # -------------------------------------------------
        if self.epoch_knots is None:
            return

        # -------------------------------------------------
        # Code that controls the schedules for digamma
        # (quantization sharpness), cycling and quantization
        # penalty...
        # -------------------------------------------------

        # HACK (mw) Crude heuristic here, make this more flexible and maybe us a learning rate scheduler class type instead

        # If we are past all cycling, set everything to the max values and return
        if epoch >= self.epoch_knots[3]:
            self.qpenalty = self.qpenalty_range[1]
            self.digamma = self.digamma_end
            return

        # If we are still at warmup, don't do anything
        if epoch <= self._warmup_epochs:
            return

        k = _knot_interval(epoch)
        cycling = k < 4 or self._last_cycle != -1

        if cycling:
            digamma_current = _digamma(k, epoch)
            if self._last_cycle == -1 and k < 4 and (k != 0 or epoch >= self._warmup_epochs + self._cycle_len):
                l = _knot_interval(epoch + self._cycle_len)
                if l < 4:
                    self._digamma_cycle_start = max(self.digamma_start, digamma_current * 0.45)
                    self._digamma_cycle_end = _digamma(l, epoch + self._cycle_len)
                    self._last_cycle = epoch
                    self.digamma = self._digamma_cycle_start
            else:
                if epoch < self._warmup_epochs + self._cycle_len:
                    self.digamma = _digamma(k, epoch)
                elif epoch <=  self._last_cycle + self._cycle_len:
                    self.digamma = _lerp(self._last_cycle, self._last_cycle + self._cycle_len, epoch, self._digamma_cycle_start, self._digamma_cycle_end)
                else:
                    self._last_cycle = -1
        else:
            self.digamma = _digamma(k, epoch)


        if k >= 2:
            self.qpenalty = _lerp(self.epoch_knots[2], self.epoch_knots[3], epoch, self.qpenalty_range[0], self.qpenalty_range[1])


    def inference(self):
        """
        Run all TMQLayers in inference mode only

        This just sets a flag which is checked by all TMQLayer instances inside the registry
        """
        self.inference_mode = True


    def register_node(self, name, node):
        """
        Register a drop-in layer replacement with the controller instance.

        :param name: Name of the instance that is registered
        :param node: Reference to the actual object that is used as quantizing drop-in replacement

        """
        self._registry[name] = node


    def clamp(self, slack=0):
        """
        Clamp the values of the weights of all registered nodes to be within the clamping range set in the constructor.

        This function iterates through all registered quantization-aware layers and instructs them to clamp their
        weights.
        """
        clamp_range = self.clamp_values[0] - slack, self.clamp_values[1] + slack
        for node in self._registry.values():
            node.clamp(clamp_range)


    def collect_grad_norms(self):
        # TODO (mw) docs
        norms = {}
        for name, node in self._registry.items():
            if node.weight.grad is not None:
                norms[name] = node.weight.grad.norm()
        return norms


    def quantize(self):
        """
        Run hard-quantization on the weights of all registered nodes to fully conform to the quantization range supplied in the controller.

        This function iterates through all registered quantization-aware layers and instructs them to hard-quantize their weights.
        You might want to run clamp() before running this function.

        Use this as a final when no further training is planned. Do not use this for checkpointing.
        """
        for node in self._registry.values():
            w = torch.round(SoftStep.map(node.weight.data, self))
            node.weight.data = w


    def quantization_loss(self, penalty=None):
        """
        Compute loss function that penalizes imperfect quantization.

        Use this method to compute a loss function (weighted by the supplied penalty) which penalizes weight data
        not being integer values.
        """
        total_loss = None
        p = penalty if penalty is not None else self.qpenalty
        for node in self._registry.values():
            nloss = node.quantization_loss(p, self.qpenalty_sharpness)
            total_loss = nloss if total_loss is None else total_loss + nloss
        return total_loss




class SoftStep(Function):
    """
    Main facility for quantization aware learning, transfer function that can be used inside PyTorch's autograd

    This function represents a staircase function which is fully differentiable and is able to interpolate from
    a standard linear mapping (1:1) to something very close to an actual staircase. When applied to the
    weights, this "traps" the weights on the staircase levels, effectively quantization them.

    The underlying function here is a piecewise polynomial, which usually computes faster than functions using
    tanh or exp functions, as those primitives are executed on the SFUs on GPUs which are not as plenty as
    standard ALUs (ranges from 1:4 to 1:8 on Turing and Ampere, tendency shifting more towards ALUs).
    """

    @staticmethod
    @torch.no_grad()
    def map(weight: torch.Tensor, ctrl:QuantizationControl) -> torch.Tensor:
        """
        Map an unquantized tensor to a soft-quantized one, used for logging / debugging and not part of the evalution chain
        """
        assert ctrl, "Control instance is required"
        fw = torch.floor(weight.data)
        arg = ctrl.digamma * weight.data - ctrl.digamma * fw - ctrl.digamma / 2.0
        val = SoftStep.sigmoid(arg) / (2.0 * SoftStep.scalar_sigmoid(ctrl.digamma / 2)) + 0.5 + fw
        return val


    @staticmethod
    @torch.no_grad()
    def dmap(weight: torch.Tensor, ctrl:QuantizationControl) -> torch.Tensor:
        """
        Map an unquantized tensor to the derivative of a soft-quantized one, used for logging / debugging and not part of the evaluation chain
        """
        assert ctrl, "Control instance is required"
        digamma = ctrl.digamma
        arg = weight.data * digamma - digamma * torch.floor(weight.data) - digamma / 2
        return digamma * SoftStep.dsigmoid(arg) / (2.0 * SoftStep.scalar_sigmoid(digamma / 2))


    @staticmethod
    def sigmoid(input: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the underlying sigmoid function for the soft-quantization
        """
        x = torch.clamp(input, -3.0, 3.0)
        x2 = x * x
        return x * (x2 + 27.0) / (27.0 + x2 * 9.0)


    @staticmethod
    def dsigmoid(input: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the 1st derivative underlying sigmoid function for the soft-quantization
        """
        x = torch.clamp(input, -3.0, 3.0)
        x2 = x * x
        return ((x2-9)*(x2-9)) / (9 * (3+x2)*(3+x2))


    @staticmethod
    def d2sigmoid(input: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the 2nd derivative underlying sigmoid function for the soft-quantization
        """
        x = torch.clamp(input, -3.0, 3.0)
        x2 = x * x
        a = (x2+3)
        return 16*x*(x2-9) / (3*a*a*a)


    @staticmethod
    def scalar_sigmoid(input: float) -> float:
        """
        Evaluate the underlying sigmoid function for a single scalar (to compute constant factors)
        """
        x = min(3.0, max(-3.0, input))
        x2 = x * x
        return x * (x2 + 27.0) / (27.0 + x2 * 9.0)


    @staticmethod
    def scalar_dsigmoid(input: float) -> float:
        """
        Evaluate the 1st derivative underlying sigmoid function for a single scalar (to compute constant factors)
        """
        x = min(3.0, max(-3.0, input))
        x2 = x * x
        return ((x2-9)*(x2-9)) / (9 * (3+x2)*(3+x2))


    @staticmethod
    def forward(ctx: FunctionCtx, weight: nn.Parameter, scale: float, ctrl: QuantizationControl):
        """
        Apply the soft quantization function to an incoming tensor within PyTorch's autograd framework

        If the CUDA helpers are compiled, it will use an accelerated CUDA kernel (gain is not a lot though)
        """
        ctx.ctrl = ctrl
        ctx.scale = scale               # NOTE (mw) scale is not used for the forward computation here, but stored for (optional) gradient scaling later
        ctx.save_for_backward(weight)
        if weight.device.type == "cuda" and CUDA_EXT_AVAILABLE:
            return fwd_softstep(weight.data, ctrl.digamma)
        else:
            fw = torch.floor(weight)
            arg = ctrl.digamma * (weight - fw) - ctrl.digamma / 2.0
            return SoftStep.sigmoid(arg) / (2.0 * SoftStep.scalar_sigmoid(ctrl.digamma / 2.0)) + 0.5 + fw


    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: torch.Tensor):
        """
        Apply chain-rule for backpropagation of gradients within PyTorch's autograd framework

        If the CUDA helpers are compiled, it will use an accelerated CUDA kernel (gain is not a lot though)
        """
        digamma = ctx.ctrl.digamma
        weight = ctx.saved_tensors[0]
        if weight.device.type == "cuda" and CUDA_EXT_AVAILABLE:
            return bwd_softstep(weight.data, digamma, ctx.scale, grad_output), None, None
        else:
            frac = weight - torch.floor(weight)
            arg = digamma * frac - digamma / 2
            grad = digamma * SoftStep.dsigmoid(arg) / (2.0 * SoftStep.scalar_sigmoid(digamma / 2.0))
            gradient = ctx.scale * grad_output * grad
            return gradient, None, None


class SoftStepDerivative(Function):
    """
    Derivative of soft-quantization function  for quantization aware learning, can be used inside PyTorch's autograd

    This function represents the 1st derivative of a staircase function which is fully differentiable and is able
    to interpolate from a standard linear mapping (1:1) to something very close to an actual staircase. When applied to
    the weights, this "traps" the weights on the staircase levels, effectively quantization them.

    The underlying function here is a piecewise polynomial, which usually computes faster than functions using
    tanh or exp functions, as those primitives are executed on the SFUs on GPUs which are not as plenty as
    standard ALUs (ranges from 1:4 to 1:8 on Turing and Ampere, tendency shifting more towards ALUs).

    Currently, this derivative is used for penalizing weights that are not properly quantized.
    """

    @staticmethod
    def forward(ctx: FunctionCtx, weight: nn.Parameter, digamma, strength):
        """
        Apply the soft quantization function to an incoming tensor within PyTorch's autograd framework

        If the CUDA helpers are compiled, it will use an accelerated CUDA kernel (gain is not a lot though)
        """
        ctx.save_for_backward(weight)
        ctx.digamma = digamma
        ctx.strength = strength
        if weight.device.type == "cuda" and CUDA_EXT_AVAILABLE:
            return fwd_softstep_derivative(weight.data, digamma, strength)
        else:
            fw = torch.floor(weight)
            frac = weight-fw
            arg = digamma * frac - digamma / 2
            output = strength * digamma * SoftStep.dsigmoid(arg) / (2.0 * SoftStep.scalar_sigmoid(digamma / 2.0))
            return output

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: torch.Tensor):
        """
        Apply chain-rule for backpropagation of gradients within PyTorch's autograd framework

        If the CUDA helpers are compiled, it will use an accelerated CUDA kernel (gain is not a lot though)
        """
        digamma = ctx.digamma
        strength = ctx.strength
        weight = ctx.saved_tensors[0]
        if weight.device.type == "cuda" and CUDA_EXT_AVAILABLE:
            return bwd_softstep_derivative(weight.data, digamma, strength, grad_output), None, None
        else:
            fw = torch.floor(weight)
            frac = weight-fw
            arg = digamma * frac - digamma / 2
            grad = strength * digamma * digamma * SoftStep.d2sigmoid(arg) / (2.0 * SoftStep.scalar_sigmoid(digamma / 2.0))
            gradient = grad_output * grad
            return gradient, None, None


class TMQLayer:
    """
    Base class for all quantization-aware layers in this project.

    Defines a basic interface and has some bookkeeping functionality.
    """
    def __init__(self, ctrl=None, quant_mode=QuantMode.NONE, tensor_type=TensorType.FLOAT_32BIT, device=None):
        self._ctrl = ctrl
        self.quantization = quant_mode
        self.tensor_type = tensor_type
        self._device = device
        self.compact = False
        self.original_shape = None
        self.post_scale_imm = 1.0
        self.post_scale = None
        self.layer_name = None
        self._scale_dirty = True

    def compact_inference_mode(self):
        """
        Switch layer into compact inference mode
        """
        self.training = False
        self.compact = True


    def quantization_loss(self, strength, sharpness):
        """
        Compute quantization loss function, to be implemented in derived classes
        """
        raise Exception("Implement in derived classes")


    def quantize(self):
        """
        Run a hard quantization on the weights (not biases) of the layer, to be implemented in derived classes
        """
        raise Exception("Implement in derived classes")


    @torch.no_grad()
    def clamp(self, limits=(-1., 1.)):
        """
        Clamp the weights (not biases) to the provided limits, to be implemented in derived classes
        """
        raise Exception("Implement in derived classes")


    def _update_scale(self):
        if self._scale_dirty and self.post_scale is not None:
            self.post_scale_imm = self.post_scale.item()
            self._scale_dirty = False


    def _variance_estimate(self) -> float:
        """
        Estimate variance of output data when routed through linear combinations of weights under different
        quantization levels. The assumption is that the input data is of unit variance without any bias.
        """
        if self.quantization == QuantMode.TERNARY:          # {-1, 0, 1}
            return 2.0/3.0
        elif self.quantization == QuantMode.TWO_BIT:        # {-2,-1,0,1}
            return (2 * 2 * 2 + 2 + 1) / 6.0
        elif self.quantization == QuantMode.THREE_BIT:      # {-4,-1,0,1}
            return (2 * 4 * 4 + 4 + 1) / 6.0
        elif self.quantization == QuantMode.FOUR_BIT:       # {-8,...7}
            return (2 * 8 * 8 + 8 + 1) / 6.0
        elif self.quantization == QuantMode.FIVE_BIT:       # {-16,...15}
            return (2 * 16 * 16 + 16 + 1) / 6.0
        elif self.quantization == QuantMode.SIX_BIT:        # {-32,...31}
            return (2 * 32 * 32 + 32 + 1) / 6.0
        elif self.quantization == QuantMode.EIGHT_BIT:      # {-128..127}
            return (2 * 128 * 128 + 128 + 1) / 6.0
        else:
            return 1.0


class Linear(nn.Module, TMQLayer):
    """
    Drop-in replacement for a standard Linear layer that is quantization aware.


    """
    def __init__(self, in_features: int, out_features: int, bias:bool=True, device=None,
                 ctrl: QuantizationControl=None, quant_mode=QuantMode.NONE, post_scale=False):
        super(Linear, self).__init__()
        TMQLayer.__init__(self, ctrl, quant_mode, TensorType.FLOAT_32BIT, device)
        self.weight = nn.Parameter()
        self._in_features = in_features
        self._out_features = out_features
        self._init_weights(bias, device)
        self.original_shape = self.weight.shape
        self.post_scale_imm = 1.0 / (self._in_features * self._variance_estimate()) if post_scale else 1.0
        self.post_scale = nn.Parameter(torch.tensor(self.post_scale_imm), requires_grad=False)
        self.bn = None


    def __repr__(self):
        if len(self.original_shape) == 2:
            return "Linear(%d,%d) [tmq]" % self.original_shape + " <%s>" % self.layer_name
        else:
            return "Linear(???) [tmq] <%s>" % self.layer_name

    def quantize(self):
        """
        Quantize weight data (FP32) to the nearest integers
        """
        self.weight.data = torch.round(self.weight.data).float()    # NOTE (mw) float for now

    def disable_bias(self):
        """
        Explicitly disable the affine component of the transform in this layer (no bias)
        """
        self.bias = None

    def implicit_bn(self):
        """
        Add an internal BatchNorm layer to this layer in order to rescale results

        A side effect of having a strict ternary set of weights is that the results of this layer are scaled
        to rather large values. In order to maintain well-behaved intermediary distributions in the tensors,
        a batch-norm layer is crucial to keep the data within an unbiased unit variance.

        Note that this will alter the model structure and this function must also be called prior to
        populating this model with checkpoint / serialized data.
        """
        self.bn = nn.BatchNorm1d(self._out_features)

    def forward(self, x):
        # mwmw
        y = self.forward_wrap(x)
        cpux = x.detach().cpu().numpy()
        cpu = y.detach().cpu().numpy()
        if self.compact:
            cpux.tofile("/tmp/ter_%s_input.bin" % self.layer_name.lstrip("."))
            cpu.tofile("/tmp/ter_%s.bin" % self.layer_name.lstrip("."))
        else:
            cpux.tofile("/tmp/ref_%s_input.bin" % self.layer_name.lstrip("."))
            cpu.tofile("/tmp/ref_%s.bin" % self.layer_name.lstrip("."))
        return y

    def forward_wrap(self, x):
        """
        Apply the layer on input data

        Same as the original forward() function in PyTorch, differentiates between "training" and "inference" mode,
        when actual compact/ternary data is loaded into the model. Will then use the compact_forward() function
        instead.
        """
        if self.compact:
            return self.compact_forward(x)
        else:
            assert self._ctrl or not self.training, "Quantization controller required for training"
            self._update_scale()
            w = SoftStep.apply(self.weight, 1.0, self._ctrl) if self._ctrl is not None and not self._ctrl.inference_mode else self.weight   # mwmw inject scale here if gradient scaling is on
            if self.post_scale != 1.0:
                if self.bias is not None:
                    final = self.bias + self.post_scale * torch.nn.functional.linear(x, w, None)
                else:
                    final = self.post_scale * torch.nn.functional.linear(x, w, None)
            else:
               final = torch.nn.functional.linear(x, w, self.bias)

        return self.bn(final) if self.bn is not None else final


    @torch.no_grad()
    def compact_forward(self, x):
        """
        Specialized forward() that works in inference only with compact data loaded into the layer.

        When this layer is populated with compact data representation (2-bits for ternary)
        """
        assert self.training == False, "Compact inference is not supported in training mode"
        assert self.weight.data.dtype == torch.int32, "Compact inference requires 32-bit integer words"
        assert CUDA_EXT_AVAILABLE, "Compact inference mode requires CUDA extension to be available"

        self._update_scale()
        rows = 1
        if len(x.shape) == 3:
            bs, rows, cols = x.shape
        elif len(x.shape) == 4:
            assert x.shape[1] == 1, "Illegal shape supplied"
            bs, _, rows, cols = x.shape
        if rows == 1:
            out = ternary_mvm_cuda(x, self.weight.data, self.original_shape[0])
        else:
            out = ternary_mmm_cuda(x, self.weight.data)
        if self.post_scale != 1.0:
            out *= self.post_scale
        if self.bias is not None:
            out += self.bias
        return self.bn(out) if self.bn is not None else out


    @torch.no_grad()
    def clamp(self, limits=(-1., 1.)):
        """
        Clamp the weights (not biases) to the provided limits, to be implemented in derived classes
        """
        self.weight.data = torch.clamp(self.weight.data, limits[0], limits[1])

    def quantization_loss(self, strength, sharpness):
        return torch.mean(SoftStepDerivative.apply(self.weight, sharpness, strength))


    @classmethod
    def from_linear(cls, source: nn.Linear, name: str, ctrl: QuantizationControl):
        # TODO (mw) docs
        assert ctrl, "Control instance is required"
        obj = cls(source.in_features, source.out_features, source.bias is not None,
                  ctrl=ctrl, device=source.weight.device, quant_mode=ctrl.quantization,
                  post_scale=ctrl.post_scale)
        obj.layer_name = name
        ctrl.register_node(name, obj)
        return obj


    @torch.no_grad()
    def _init_weights(self, bias, device):
        # TODO (mw) docs
        if bias:
            b = torch.zeros(self._out_features)
            self.bias = nn.Parameter(b.to(device), requires_grad=True)
        else:
            self.bias = None
        # TODO (mw) configurable weight initialization range, for now we go -1..1
        scale = 2.0
        #scale = 2.0 * math.sqrt(6.0/(self._out_features + self._in_features))       # Xavier init (too narrow ?) TODO (mw) double-check
        w = scale * (torch.rand(self._out_features, self._in_features) - 0.5)
        self.weight = nn.Parameter(w.to(device), requires_grad=True)


    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """
        Hacky overload of a PyTorch private state loading routine

        This intercepts the standard data loading functionality in PyTorch on a low level and is very hacky.
        Unfortunately the default implementation of load_state_dict in the Module class recursively descends
        the network and instead of using a public method like load_state_dict() to load the state dictionary
        for each dependent module, it descends using the internal _load_from_state_dict() function on itself
        and its children.

        :param state_dict: State dictionary for this module with ONLY the data for this module
        :param prefix: Prefix of the module hierarchy excluding the name of this module
        :param local_metadata: Metadata for this module (if any)
        :param strict: Whether to apply strict data loading
        :param missing_keys:
        :param unexpected_keys:
        :param error_msgs:
        """
        super(Linear, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        if "tmq.quant" in local_metadata and "tmq.tensor_type" in local_metadata:
            self.tensor_type = TensorType(local_metadata["tmq.tensor_type"])
            self.original_shape = local_metadata["tmq.shape"]
            if self.tensor_type == TensorType.COMPRESSED_TERNARY_32BIT:
                if not NATIVE_EXT_AVAILABLE:
                    raise Exception("Native extension required to use entropy-coded weight data")
                self.weight = nn.Parameter(decompress_ternary(state_dict[prefix + "weight"], self.original_shape), requires_grad=False)
                self.tensor_type = TensorType.COMPACT_TERNARY_32BIT
                self.compact = True
            elif self.tensor_type == TensorType.COMPACT_TERNARY_32BIT:
                self.weight = nn.Parameter(state_dict[prefix + "weight"], requires_grad=False)
                self.compact = True
            else:
                self.compact = False
            self.quantization = QuantMode(local_metadata["tmq.quant"])
            self.training = False if self.compact else self.training
        else:
            self.compact = False            # This is just a standard floating-point data load



class Conv2d(nn.Module, TMQLayer):
    """
    In-training quantization version of a 2D convolution layer

    This layer can be used as a drop-in replacement for PyTorch's standard Conv2d layer. It provides the same
    interface as the original and extends it by a few functions that handle the quantization part.
    """

    def __init__(self, in_channels, out_channels, kernel, stride=1, padding=0, groups=1, dilation=1,
                 bias=True, device=None, ctrl=None, quant_mode=QuantMode.NONE, post_scale=False):
        super(Conv2d, self).__init__()
        TMQLayer.__init__(self, ctrl, quant_mode, TensorType.FLOAT_32BIT, device)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilation = dilation
        self._init_weights(bias, device)
        self.original_shape = self.weight.shape
        self.post_scale_imm = 1.0 / (self._fan_in * self._variance_estimate()) if post_scale else 1.0
        self.post_scale = nn.Parameter(torch.tensor(self.post_scale_imm), requires_grad=False)


    def __repr__(self):
        return "Conv2d(%d,%d,%d,%d) [tmq]" % self.original_shape + " <%s>" % self.layer_name


    def to(self, *args, **kwargs):
        # TODO (mw) docs
        me = super().to(*args, **kwargs)
        me.post_scale_imm = me.post_scale.item()
        return me


    def forward(self, x):
        # TODO (mw) docs
        if self.compact:
            return self.compact_forward(x)
        else:
            assert self._ctrl or not self.training, "Quantization controller required for training"
            self._update_scale()
            w = SoftStep.apply(self.weight, 1.0, self._ctrl) if self._ctrl is not None and not self._ctrl.inference_mode else self.weight
            if self.post_scale != 1.0:
                if self.bias is not None:
                    return self.bias + self.post_scale * torch.nn.functional.conv2d(x, w, None, self.stride, self.padding, dilation=self.dilation, groups=self.groups)
                else:
                    return self.post_scale * torch.nn.functional.conv2d(x, w, None, self.stride, self.padding, dilation=self.dilation, groups=self.groups)
            else:
                return torch.nn.functional.conv2d(x, w, self.bias, self.stride, self.padding, dilation=self.dilation, groups=self.groups)


    @torch.no_grad()
    def compact_forward(self, x):
        # TODO (mw) docs
        assert not self.training, "Compact inference is not supported in training mode"
        assert self.weight.data.dtype == torch.int32, "Compact inference requires 32-bit integer words"
        assert CUDA_EXT_AVAILABLE, "Compact inference mode requires CUDA extension to be available"

        self._update_scale()
        bs = 1
        if len(x.shape) == 4:
            bs, chan, h, w = x.shape
        elif len(x.shape) == 3:
            chan, h, w = x.shape
        else:
            raise Exception("Expected a 3D tensor")

        pady, padx = self.padding if isinstance(self.padding, tuple) else (self.padding, self.padding)
        diy, dix = self.dilation if isinstance(self.dilation, tuple) else (self.dilation, self.dilation)
        ky, kx = self.kernel if isinstance(self.kernel, tuple) else (self.kernel, self.kernel)
        stridey, stridex = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
        h_eff = h + 2 * pady - (diy - 1)
        w_eff = w + 2 * padx - (dix - 1)
        h_out = (h_eff - (ky - 1)) // stridey
        w_out = (w_eff - (kx - 1)) // stridex

        assert self.groups == 1 or self.groups == self.original_shape[0], "Grouped convolutions are only supported for group == channels"

        if self.groups != 1:
            unfolded = torch.nn.functional.unfold(x, kernel_size=(ky,kx), dilation=(diy,dix), stride=(stridey, stridex), padding=(pady, padx))
            tmp = unfolded.view(bs, chan, ky, kx, -1)
            permuted = tmp.permute(0,4,2,3,1).reshape(bs, h*w, ky*kx*chan).contiguous()
            convolved = ternary_dwconv_cuda(permuted, self.weight, ky, kx, chan)
            final = torch.nn.functional.fold(convolved.transpose(1,2), output_size=(h_out, w_out), kernel_size=1)

        else:
            unfolded = torch.nn.functional.unfold(x, kernel_size=(ky, kx), dilation=(diy, dix), stride=(stridey, stridey), padding=(pady, padx))
            lhs = unfolded.permute(1,0).contiguous() if len(unfolded.shape) < 3 else unfolded.permute(0,2,1).contiguous()
            out = ternary_mmm_cuda(lhs, self.weight.data.view(-1, self.weight.data.shape[-1]), self.original_shape[0])
            if len(out.shape) == 2:
                final = torch.nn.functional.fold(out.transpose(0, 1), output_size=(h_out, w_out), kernel_size=1)
            else:
                final = torch.nn.functional.fold(out.transpose(1, 2), output_size=(h_out, w_out), kernel_size=1)

        if self.post_scale != 1.0:
            final *= self.post_scale

        if self.bias is not None:
            return self.bias.view(1, self.bias.shape[0], 1, 1) + final
        else:
            return final


    @torch.no_grad()
    def clamp(self, limits=(-1., 1.)):
        """
        Clamp the weights (not biases) to the provided limits, to be implemented in derived classes
        """
        self.weight.data = torch.clamp(self.weight.data, limits[0], limits[1])


    def quantize(self):
        """
        Run a hard quantization on the weights (not biases) of the layer, to be implemented in derived classes
        """
        self.weight.data = torch.round(self.weight.data).float()        # NOTE (mw) float for now


    def quantization_loss(self, strength, sharpness):
        # TODO (mw) docs
        return torch.mean(SoftStepDerivative.apply(self.weight, sharpness, strength))


    @classmethod
    def from_conv2d(cls, source: nn.Conv2d, name: str, ctrl: QuantizationControl):
        # TODO (mw) docs
        assert ctrl, "Controller required"
        obj = cls(source.in_channels, source.out_channels, source.kernel_size,
                  stride=source.stride,
                  padding=source.padding,
                  groups=source.groups,
                  dilation=source.dilation,
                  bias=source.bias is not None,
                  device=source.weight.device,
                  ctrl=ctrl,
                  quant_mode=ctrl.quantization,
                  post_scale=ctrl.post_scale)
        obj.layer_name = name
        ctrl.register_node(name, obj)
        return obj


    @torch.no_grad()
    def _init_weights(self, bias, device):
        # TODO (mw) allow for initialization range on the weights
        if isinstance(self.kernel, tuple):
            self._fan_in = self.kernel[0] * self.kernel[1] * self.in_channels
            self._fan_out = self.kernel[0] * self.kernel[1] * self.out_channels
            scale = 2.0
            #scale = 2.0 * math.sqrt(6.0/(self._fan_in * self._fan_out))         # Xavier init (too narrow ?) TODO (mw) double-check
            wdata = scale * (torch.rand([self.out_channels, self.in_channels // self.groups, self.kernel[0], self.kernel[1]]) - 0.5)
        else:
            self._fan_in = self.kernel * self.kernel * self.in_channels
            self._fan_out = self.kernel * self.kernel * self.out_channels
            scale = 2.0
            #scale = 2.0 * math.sqrt(6.0/(self._fan_in * self._fan_out))        # Xavier init (too narrow ?) TODO (mw) double-check
            wdata = scale * (torch.rand([self.out_channels , self.in_channels // self.groups, self.kernel, self.kernel]) - 0.5)

        self.weight = nn.Parameter(wdata.to(device), requires_grad=True)

        if bias:
            bdata = torch.zeros(self.out_channels)
            self.bias = nn.Parameter(bdata.to(device), requires_grad=True)
        else:
            self.bias = None

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """
        Hacky overload of a PyTorch private state loading routine

        This intercepts the standard data loading functionality in PyTorch on a low level and is very hacky.
        Unfortunately the default implementation of load_state_dict in the Module class recursively descends
        the network and instead of using a public method like load_state_dict() to load the state dictionary
        for each dependent module, it descends using the internal _load_from_state_dict() function on itself
        and its children.

        :param state_dict: State dictionary for this module with ONLY the data for this module
        :param prefix: Prefix of the module hierarchy excluding the name of this module
        :param local_metadata: Metadata for this module (if any)
        :param strict: Whether to apply strict data loading
        :param missing_keys:
        :param unexpected_keys:
        :param error_msgs:
        """
        super(Conv2d, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        if "tmq.quant" in local_metadata and "tmq.tensor_type" in local_metadata:
            self.tensor_type = TensorType(local_metadata["tmq.tensor_type"])
            self.original_shape = local_metadata["tmq.shape"]
            if self.tensor_type == TensorType.COMPRESSED_TERNARY_32BIT:
                if not NATIVE_EXT_AVAILABLE:
                    raise Exception("Native extension required to use entropy-coded weight data")
                self.weight = nn.Parameter(decompress_ternary(state_dict[prefix + "weight"], self.original_shape), requires_grad=False)
                self.tensor_type = TensorType.COMPACT_TERNARY_32BIT
                self.compact = True
            elif self.tensor_type == TensorType.COMPACT_TERNARY_32BIT:
                self.weight = nn.Parameter(state_dict[prefix + "weight"], requires_grad=False)
                self.compact = True
            else:
                self.compact = False
            self.training = False if self.compact else self.training
            self.quantization = QuantMode(local_metadata["tmq.quant"])
        else:
            self.compact = False




class ConvTranspose2d(Conv2d):

    def __init__(self, in_channels, out_channels, kernel, stride=1, padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1, device=None, ctrl=None, quant_mode=QuantMode.NONE, post_scale=False):
        super(ConvTranspose2d, self).__init__(in_channels, out_channels, kernel, stride, padding, groups, dilation, bias, device, ctrl,
                                              quant_mode, post_scale)
        self.output_padding = output_padding

    def __repr__(self):
        return "ConvTranspose2d(%d,%d,%d,%d) [tmq] " % self.original_shape + " <%s>" % self.layer_name


    def forward(self, x):
        # TODO (mw) docs
        if self.compact:
            return self.compact_forward(x)
        else:
            assert self._ctrl or not self.training, "Quantization controller required for training"
            self._update_scale()
            w = SoftStep.apply(self.weight, 1.0, self._ctrl) if self._ctrl is not None and not self._ctrl.inference_mode else self.weight
            if self.post_scale != 1.0:
                if self.bias is not None:
                    return self.bias + self.post_scale * torch.nn.functional.conv_transpose2d(x, w, None, self.stride, self.padding, dilation=self.dilation, groups=self.groups)
                else:
                    return self.post_scale * torch.nn.functional.conv_transpose2d(x, w, None, self.stride, self.padding, dilation=self.dilation, groups=self.groups)
            else:
                return torch.nn.functional.conv_transpose2d(x, w, self.bias, self.stride, self.padding, dilation=self.dilation, groups=self.groups)


    @torch.no_grad()
    def compact_forward(self, x):
        assert self.training == False, "Compact inference is not supported in training mode"
        assert self.weight.data.dtype == torch.int32, "Compact inference requires 32-bit integer words"
        assert CUDA_EXT_AVAILABLE, "Compact inference mode requires CUDA extension to be available"

        self._update_scale()
        pady, padx = self.padding if isinstance(self.padding, tuple) else (self.padding, self.padding)
        opady, opadx = self.padding if isinstance(self.output_padding, tuple) else (self.output_padding, self.output_padding)
        diy, dix = self.dilation if isinstance(self.dilation, tuple) else (self.dilation, self.dilation)
        ky, kx = self.kernel if isinstance(self.kernel, tuple) else (self.kernel, self.kernel)
        stridey, stridex = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
        h, w = x.shape[-2:]
        h_out = (h - 1) * stridey - 2 * pady + (diy * (ky - 1) + 1) + opady
        w_out = (w - 1) * stridex - 2 * padx + (dix * (kx - 1) + 1) + opadx
        bs, chan, h, w = x.shape

        assert self.groups == 1 or self.groups == self.in_channels, "Grouped transpose convolutions are only supported for channels == groupsize"

        if self.groups != 1:
            tmp = ternary_dwconvtrans_cuda(x.reshape(bs, chan, h*w).transpose(1,2).contiguous(), self.weight.data, ky, kx)
            final = torch.nn.functional.fold(tmp, output_size=(h_out, w_out), kernel_size=(ky,kx),
                                             dilation=(diy,dix), stride=(stridey,stridex), padding=(pady, padx))
        else:
            lhs = x.permute(0, 2, 3, 1).reshape(bs, h * w, chan)
            out = ternary_mmm_cuda(lhs, self.weight.data.view(self.weight.data.shape[0],-1), self.original_shape[0])
            final = torch.nn.functional.fold(out.permute(0, 2, 1), output_size=(h_out, w_out), kernel_size=(ky, kx),
                                             dilation=(diy, dix), stride=(stridey, stridex), padding=(pady, padx))

        if self.post_scale != 1.0:
            final *= self.post_scale

        if self.bias is not None:
            return self.bias.view(1, self.bias.shape[0], 1, 1) + final
        else:
            return final


    @classmethod
    def from_transconv2d(cls, source: nn.ConvTranspose2d, name: str, ctrl: QuantizationControl):
        # TODO (mw) docs
        assert ctrl, "Controller required"
        obj = cls(source.in_channels, source.out_channels, source.kernel_size,
                  stride=source.stride,
                  padding=source.padding,
                  output_padding=source.output_padding,
                  groups=source.groups,
                  dilation=source.dilation,
                  bias=source.bias is not None,
                  device=ctrl.device,
                  ctrl=ctrl,
                  quant_mode=ctrl.quantization,
                  post_scale=ctrl.post_scale)
        obj.layer_name = name
        ctrl.register_node(name, obj)
        return obj


    @torch.no_grad()
    def _init_weights(self, bias, device):
        # TODO (mw) docs
        if isinstance(self.kernel, tuple):
            self._fan_in = self.kernel[0] * self.kernel[1] * self.in_channels
            self._fan_out = self.kernel[0] * self.kernel[1] * self.out_channels
            scale = 2.0
            #scale = 2.0 * math.sqrt(6.0/(self._fan_in * self._fan_out))         # Xavier init (too narrow ?) TODO (mw) double-check
            wdata = scale * (torch.rand([self.in_channels, self.out_channels // self.groups, self.kernel[0], self.kernel[1]]) - 0.5)
            self._fan_in = self.kernel[0] * self.kernel[1] * self.in_channels
        else:
            self._fan_in = self.kernel * self.kernel * self.in_channels
            self._fan_out = self.kernel * self.kernel * self.out_channels
            scale = 2.0
            #scale = 2.0 * math.sqrt(6.0/(self._fan_in * self._fan_out))         # Xavier init (too narrow ?) TODO (mw) double-check
            wdata = scale * (torch.rand([self.in_channels, self.out_channels // self.groups, self.kernel, self.kernel]) - 0.5)

        self.weight = nn.Parameter(wdata.to(device), requires_grad=True)
        if bias:
            bdata = torch.zeros(self.out_channels)
            self.bias = nn.Parameter(bdata.to(device), requires_grad=True)
        else:
            self.bias = None


def quantize_weights(module: nn.Module):
    """
    Quantize/snap the weights of a TMQ-layer model to quantized values

    This function runs an in-place quantization of all model weights that are part of TMQ layers.

    :param module: Top-level module (the whole model) to run quantization on
    """
    _recursive_quantize_weights(module)


def _recursive_quantize_weights(module: nn.Module):
    # TODO (mw) docs
    for name, sub in module.named_children():
        if isinstance(sub, TMQLayer):
            if sub.weight.data.dtype == torch.float16:
                raise Exception("FP16 data is currently not supported")
            elif sub.weight.data.dtype == torch.float32:
                rq = sub.weight.requires_grad
                sub.weight = nn.Parameter(torch.round(sub.weight.data), requires_grad = rq)
            # we assume that any non-fp weight is already in compact form
        _recursive_quantize_weights(sub)


def quantize_model(module: nn.Module, ctrl: QuantizationControl, prefix="", exclude=[]):
    """
    Perform an in-place swap of model/module components in favour of quantization-aware layers

    This function traverses the supplied module and checks for layer types which can be swapped for a layer
    that performs in-training quantization. Those layers are swapped in-place, altering the supplied
    module. In order to provide some control in terms of which layers should be replaced, layers can
    be excluded on a per name basis.

    For example, assuming a prefix of "model" is supplied and the top-level module contains a submodule named
    "conv1" which should not be quantized, add "model.conv1" to the exclusion list supplied in the exclude
    parameter. The exclusion list supports basic wildcard matching/globbing using the "*" symbol. For
    example, if all layers that are named "conv5" are to be excluded, supplying "*conv5" will take care of that.

    :param module: Top-level module (the whole model) to run quantization on
    :param ctrl: Quantization control instance
    :param prefix: The initial prefix (name) to assign to the top-level module, usually "model" is a good choice here.
    :param exclude: Optional list of layer names and/or wildcards that should be excluded from quantization
    """
    assert ctrl, "Must supply a quantization control instance"
    ex, wildcards = [], []
    for phrase in exclude:
        if "*" in phrase:
            wildcards.append(phrase)
        else:
            ex.append(phrase)
    _recursive_quantize_module(module, ctrl, prefix, ex, wildcards)



def _recursive_quantize_module(module: nn.Module, ctrl: QuantizationControl, prefix, exclude, wc_exclude):
    """
    Recursively "quantize" a model in a module-by-module fashion.

    This function checks if a module can be replaced by its quantized version (and also if it should be replaced) and
    then performs an in-place replacement.

    This step should be done right after creating the model, as all model parameters will be lost and initialized
    to different values.

    :param module: Module to check for replacement
    :param ctrl: Quantization control instance
    :param prefix: Module prefix string (hierarchy of module names separated by dots)
    :param exclude: List of names that should be excluded from quantization
    :param wc_exclude: List of wildcard expressions that should be excluded from quantization
    """
    for name, sub in module.named_children():
        if sub != module:
            newpre = prefix + "." + name if prefix != "" else name
            if newpre not in exclude:
                skip = False
                if len(wc_exclude) > 0:
                    for phrase in wc_exclude:
                        if len(fnmatch.filter([newpre], phrase)) > 0:
                            skip = True
                if not skip:
                    if isinstance(sub, nn.Linear):
                       setattr(module, name, Linear.from_linear(sub, prefix+"."+name, ctrl))
                    if isinstance(sub, nn.Conv2d):
                       setattr(module, name, Conv2d.from_conv2d(sub, prefix+"."+name, ctrl))
                    if isinstance(sub, nn.ConvTranspose2d):
                        setattr(module, name, ConvTranspose2d.from_transconv2d(sub, prefix+"."+name, ctrl))
                else:
                    print("Skipping excluded layer %s" % newpre)
                _recursive_quantize_module(sub, ctrl, newpre, exclude, wc_exclude)
            else:
                print("Skipping excluded layer %s" % newpre)


def compressed_state_dict(module: nn.Module, strong=False, remove_prefix=""):
    """
    Generate a compressed/compact version of the model state dictionary for storage.

    This function compactifies/compresses the weights of the quantized layers using either simple bit-stuffing
    or a range encoder that performs entropy coding for an even more compact storage.

    Note that as of now, we only support ternary data in compact/compressed form. Also, the supplied model is
    put into "compact inference mode" because the weights are altered in-place by this function.

    :param module: Root module (model) to generate a compressed state dictionary for
    :param strong: Flag that controls whether an entropy coder should be used instead of simple bit-stuffing

    :return: State dictionary with quantized layers storing their (weight) data in compact or even compressed form.
    """
    # TOOD (mw) support other quantization types than just ternary
    _recursive_compress_ternary(module, strong)
    state = module.state_dict()
    _recursive_state_endowment(module, state, strong)
    return state


def _recursive_compress_ternary(module: nn.Module, strong):
    """
    Recursively descend module hierarchy and compress quantized layer weights

    :param module: Module to work on
    :param strong: If set to true, will use entropy coding for the storage inside the state dictionary (it will
                   not use entropy-coded data on the weights in memory)
    """
    for name, sub in module.named_children():
        if isinstance(sub, TMQLayer):
            if hasattr(sub, "tensor_type"):
                if not hasattr(sub, "original_shape"):
                    sub.original_shape = sub.weight.shape
                if not strong and sub.tensor_type != TensorType.COMPACT_TERNARY_32BIT:
                    compressed = compactify_ternary(sub.weight.data)
                    sub.tensor_type = TensorType.COMPACT_TERNARY_32BIT
                    sub.weight = nn.Parameter(compressed, requires_grad=False)
                elif strong and sub.tensor_type != TensorType.COMPRESSED_TERNARY_32BIT:
                    if not NATIVE_EXT_AVAILABLE:
                        raise Exception("Cannot store entropy-coded weight data without native helper available")
                    compressed = compress_ternary(sub.weight.data, sub.original_shape, True if sub.tensor_type == TensorType.COMPACT_TERNARY_32BIT else False)
                    sub.tensor_type = TensorType.COMPRESSED_TERNARY_32BIT
                    sub.weight = nn.Parameter(compressed, requires_grad=False)
        else:
            _recursive_compress_ternary(sub, strong)


def _recursive_state_endowment(module: nn.Module, state_dict, strong, prefix=""):
    # TODO (mw) docs
    for name, sub in module.named_children():
        fullname = prefix + "." + name if prefix != "" else name
        if isinstance(sub, TMQLayer):
            sub.compact_inference_mode()
            meta = state_dict._metadata[fullname]
            meta["tmq.tensor_type"] = sub.tensor_type.value
            meta["tmq.shape"] = sub.original_shape
            meta["tmq.quant"] = sub.quantization.value
            if sub.tensor_type == TensorType.COMPRESSED_TERNARY_32BIT:
                # For cases where inference is to be done right after getting the compressed state dict, we
                # decompress to compact representation here
                sub.weight = nn.Parameter(decompress_ternary(sub.weight.data, sub.original_shape), requires_grad=False)
                sub.tensor_type = TensorType.COMPACT_TERNARY_32BIT
        else:
            _recursive_state_endowment(sub, state_dict, strong, fullname)


def prepare_compressed_load(module: nn.Module, state_dict):
    _recursive_cmp_model_adjustment(module, state_dict, "")


# TODO (mw) retire ?
def load_compressed_state_dict(module: nn.Module, state_dict):
    """
    Load a compressed/compacted state dictionary into the model with the quantized layers.

    This function loads a compressed or compacted state_dictionary back into the model, overwriting any existing
    parameter data. For TMQ-type layers, the layer flags are adjusted with the data. In case the data was stored
    in fully compressed form, it is first decompressed and stored in memory in the compact (fixed length) form.

    :param module: Target module (model) to load the supplied state dictionary in
    :param state_dict: State dictionary which contains all data to operate in compact form
    """
    _recursive_cmp_model_adjustment(module, state_dict)
    module.load_state_dict(state_dict, strict=False)


def _recursive_cmp_model_adjustment(module: nn.Module, state_dict, prefix=""):
    """
    Recursively parse additional data from the state dictionary that is related to the compressed/compact
    representation of the model.

    :param module: Module to check for additional attributes
    :param state_dict: State dictionary that was stored in compressed form
    :param prefix: Current module prefix (module names separated by dots)
    """
    for name, sub in module.named_children():
        fullname = prefix + "." + name if prefix != "" else name
        meta = state_dict._metadata
        if isinstance(sub, TMQLayer):
            sub.layer_name = fullname
            if fullname in meta:
                weight_name = fullname + ".weight"
                tensor_type = TensorType(meta[fullname]["tmq.tensor_type"])
                if tensor_type in [TensorType.COMPACT_TERNARY_32BIT, TensorType.COMPRESSED_TERNARY_32BIT]:
                    sub.weight = nn.Parameter(torch.empty(state_dict[weight_name].shape, dtype=state_dict[weight_name].dtype), requires_grad=False)
                else:
                    sub.compact = False
        else:
            _recursive_cmp_model_adjustment(sub, state_dict, fullname)


def expand_compact_layers(module: nn.Module, for_training=False):
    """
    Expand TMQ layers with compact layers to 32-bit floating point representation (in place)

    :param module: Target module (model) to load the supplied state dictionary in
    :param for_training: If set to True, makes sure that the "required_grad" flag is set to True on the weights,
                         otherwise it is set to False
    """
    if not NATIVE_EXT_AVAILABLE:
        raise Exception("Must have TMQ native extension compiled and available to decompress data")
    _recursive_expansion(module, for_training)


def _recursive_expansion(module: nn.Module, for_training):
    # TODO (mw) docs
    for _, sub in module.named_children():
        if isinstance(sub, TMQLayer) and hasattr(sub, "tensor_type") and hasattr(sub,"original_shape"):
            entropy = True if sub.tensor_type == TensorType.COMPRESSED_TERNARY_32BIT else False
            if sub.tensor_type == TensorType.COMPRESSED_TERNARY_32BIT or sub.tensor_type == TensorType.COMPACT_TERNARY_32BIT:
                full = expand_ternary(sub.weight.data, sub.original_shape, entropy)
                sub.weight = nn.Parameter(full, requires_grad=for_training)
                sub.tensor_type = TensorType.FLOAT_32BIT
                sub.compact = False
        _recursive_expansion(sub, for_training)

