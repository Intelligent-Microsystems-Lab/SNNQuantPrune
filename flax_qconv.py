# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Copied code from
# https://github.com/google/flax/blob/master/flax/linen/linear.py and
# https://github.com/google/jax/blob/master/jax/_src/lax/lax.py
# modified to accomodate noise and quantization

from typing import Any, Callable, Iterable, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
# import jax
from jax import lax

from jax._src.lax.convolution import (
    conv_dimension_numbers,
    #     ConvDimensionNumbers,
    #    _conv_sdims,
    #    _conv_spec_transpose,
    #    _conv_general_vjp_lhs_padding,
    #    _conv_general_vjp_rhs_padding,
    #   _reshape_axis_out_of,
    #   _reshape_axis_into,
)

from jax._src.lax.lax import (
    padtype_to_pads,
    #    rev,
)

from flax.linen.module import Module, compact
from flax.linen.linear import (
    zeros,
    default_kernel_init,
    PRNGKey,
    Shape,
    Dtype,
    Array,
    _conv_dimension_numbers,
)

from quant import prune


class QuantConv(Module):
  """Convolution Module wrapping lax.conv_general_dilated.
  Attributes:
    features: number of convolution filters.
    kernel_size: shape of the convolutional kernel. For 1D convolution,
      the kernel size can be passed as an integer. For all other cases, it
      must be a sequence of integers.
    strides: a sequence of `n` integers, representing the inter-window
      strides.
    padding: either the string `'SAME'`, the string `'VALID'`, or a sequence
      of `n` `(low, high)` integer pairs that give the padding to apply
      before and after each spatial dimension.
    input_dilation: `None`, or a sequence of `n` integers, giving the
      dilation factor to apply in each spatial dimension of `inputs`.
      Convolution with input dilation `d` is equivalent to transposed
      convolution with stride `d`.
    kernel_dilation: `None`, or a sequence of `n` integers, giving the
      dilation factor to apply in each spatial dimension of the convolution
      kernel. Convolution with kernel dilation is also known as 'atrous
      convolution'.
    feature_group_count: integer, default 1. If specified divides the input
      features into groups.
    use_bias: whether to add a bias to the output (default: True).
    dtype: the dtype of the computation (default: float32).
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    kernel_init: initializer for the convolutional kernel.
    bias_init: initializer for the bias.
    config: ???
  """

  features: int
  kernel_size: Union[int, Iterable[int]]
  strides: Optional[Iterable[int]] = None
  padding: Union[str, Iterable[Tuple[int, int]]] = "SAME"
  input_dilation: Optional[Iterable[int]] = None
  kernel_dilation: Optional[Iterable[int]] = None
  feature_group_count: int = 1
  use_bias: bool = True
  dtype: Dtype = jnp.float32
  precision: Any = None
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
  config: dict = None
  bits: int = 8
  quant_act_sign: bool = True
  g_scale: float = 0.

  @compact
  def __call__(self, inputs: Array, rng: Any = None) -> Array:
    """Applies a convolution to the inputs.
    Args:
      inputs: input data with dimensions (batch, spatial_dims..., features)
    Returns:
      The convolved data.
    """
    inputs = jnp.asarray(inputs, self.dtype)
    cfg = self.config

    if isinstance(self.kernel_size, int):
      kernel_size = (self.kernel_size,)
    else:
      kernel_size = self.kernel_size  # type: ignore

    is_single_input = False
    if inputs.ndim == len(kernel_size) + 1:
      is_single_input = True
      inputs = jnp.expand_dims(inputs, axis=0)

    strides = self.strides or (1,) * (inputs.ndim - 2)

    in_features = inputs.shape[-1]
    assert in_features % self.feature_group_count == 0
    kernel_shape = kernel_size + (
        in_features // self.feature_group_count,
        self.features,
    )
    kernel = self.param("kernel", self.kernel_init, kernel_shape)
    kernel = jnp.asarray(kernel, self.dtype)
    dimension_numbers = _conv_dimension_numbers(inputs.shape)

    dnums = conv_dimension_numbers(
        inputs.shape, kernel.shape, dimension_numbers
    )
    rhs_dilation = (1,) * (kernel.ndim - 2)
    # lhs_dilation = (1,) * (inputs.ndim - 2)
    if isinstance(self.padding, str):
      lhs_perm, rhs_perm, _ = dnums
      rhs_shape = np.take(kernel.shape, rhs_perm)[2:]
      effective_rhs_shape = [
          (k - 1) * r + 1 for k, r in zip(rhs_shape, rhs_dilation)
      ]
      padding = padtype_to_pads(
          np.take(inputs.shape, lhs_perm)[2:],
          effective_rhs_shape,
          strides,
          self.padding,
      )
    else:
      padding = self.padding

    # Quantization.
    if "weight" in cfg:
      if self.bits is not None:
        kernel_fwd = cfg.weight(bits=self.bits, g_scale=self.g_scale)(kernel)
      else:
        kernel_fwd = cfg.weight(g_scale=self.g_scale)(kernel)
    else:
      kernel_fwd = kernel

    if self.config.prune_percentage >= 0.:
      kernel_fwd = prune()(kernel_fwd)

    y = lax.conv_general_dilated(
        inputs,
        kernel_fwd,
        strides,
        padding,
        lhs_dilation=self.input_dilation,
        rhs_dilation=self.kernel_dilation,
        dimension_numbers=dimension_numbers,
        feature_group_count=self.feature_group_count,
        precision=self.precision
    )

    if is_single_input:
      y = jnp.squeeze(y, axis=0)

    if self.use_bias:
      bias = self.param("bias", self.bias_init, (self.features,))
      bias = jnp.asarray(bias, self.dtype)

      if "bias" in self.config:
        if self.bits is not None:
          bias = self.config.bias(
              bits=self.bits, g_scale=self.g_scale,
              maxabs_w=jnp.max(jnp.abs(kernel)))(bias)
        else:
          bias = self.config.bias(
              bits=self.bits, g_scale=self.g_scale,
              maxabs_w=jnp.max(jnp.abs(kernel)))(bias)

      y = y + bias
    return y
