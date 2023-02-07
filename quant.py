# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Quantization functions.

import jax
import numpy as np
import jax.numpy as jnp
from flax import linen as nn

from typing import Any, Iterable, Callable
from jax.nn.initializers import constant


Array = Any
PRNGKey = Any
Shape = Iterable[int]


def get_noise(x: Array, percentage: float, rng: PRNGKey) -> Array:
  return (
      jnp.max(jnp.abs(x)) * percentage * jax.random.uniform(
          rng, x.shape, minval=-1, maxval=1.0)
  )


@jax.custom_vjp
def round_ste(x, scale, off=False):
  return jnp.where(off, x, jnp.round(x))


def round_ste_fwd(x, scale, off=False):
  return round_ste(x, scale), (x, scale)


def round_ste_bwd(res, g):
  (x, scale) = res
  return (g, None, None)


round_ste.defvjp(round_ste_fwd, round_ste_bwd)


#
# Rounding with different backward passes
#

@jax.custom_vjp
def round_gaussian_noise(x, scale, off=False):
  return jnp.where(off, x, jnp.round(x))


def round_gaussian_noise_fwd(x, scale, off=False):
  return round_gaussian_noise(x, scale), (x, scale)


def round_gaussian_noise_bwd(res, g):
  (x, scale) = res
  key = jax.random.PRNGKey(np.random.randint(0, 100000))
  return (g * (1 + jax.random.normal(key, shape=g.shape) * scale), None, None)


round_gaussian_noise.defvjp(round_gaussian_noise_fwd, round_gaussian_noise_bwd)


@jax.custom_vjp
def round_uniform_noise(x, scale, off=False):
  return jnp.where(off, x, jnp.round(x))


def round_uniform_noise_fwd(x, scale, off=False):
  return round_uniform_noise(x, scale), (x, scale)


def round_uniform_noise_bwd(res, g):
  (x, scale) = res
  key = jax.random.PRNGKey(np.random.randint(0, 100000))
  return (g * (1 + jax.random.uniform(key, shape=g.shape, minval=-.5,
                                      maxval=.5) * scale), None, None)


round_uniform_noise.defvjp(round_uniform_noise_fwd, round_uniform_noise_bwd)

# Type 1: approximations of rounding.

# ewgs https://arxiv.org/pdf/2104.00903.pdf


@jax.custom_vjp
def round_ewgs(x, scale, off=False):
  return jnp.where(off, x, jnp.round(x))


def round_ewgs_fwd(x, scale, off=False):
  return round_ewgs(x, scale), (x, scale)


def round_ewgs_bwd(res, g):
  (x, scale) = res

  return (g * (1 + scale * jnp.sign(g) * (x - jnp.round(x))), None, None)


round_ewgs.defvjp(round_ewgs_fwd, round_ewgs_bwd)


@jax.custom_vjp
def round_acos(x, scale, off=False):
  return jnp.where(off, x, jnp.round(x))


def round_acos_fwd(x, scale, off=False):
  return round_acos(x, scale), (x, scale)


def round_acos_bwd(res, g):
  (x, scale) = res

  modulator = .5 * jnp.sin(jnp.pi * (x - jnp.round(x)))

  return (g * (1 + scale * modulator), None, None)


round_acos.defvjp(round_acos_fwd, round_acos_bwd)


@jax.custom_vjp
def round_tanh(x, scale, off=False, alpha_scale=1.):
  return jnp.where(off, x, jnp.round(x))


def round_tanh_fwd(x, scale, off=False, alpha_scale=1.):
  return round_tanh(x, scale, off=off), (x, scale, alpha_scale)


def round_tanh_bwd(res, g):
  (x, scale, alpha_scale) = res

  # a parameter to scale the softness/steepness.
  alpha = 4
  tanh_coeff = (1 + scale * .5 * jnp.sign(g) * jax.nn.tanh(
      (x - jnp.round(x)) * alpha))
  ewgs_coeff = (1 + scale * jnp.sign(g) * (x - jnp.round(x)))
  return (g * (tanh_coeff * alpha_scale + ewgs_coeff * (1 - alpha_scale)),
          None, None, None)


round_tanh.defvjp(round_tanh_fwd, round_tanh_bwd)


@jax.custom_vjp
def round_invtanh(x, scale, off=False, alpha_scale=1.):
  return jnp.where(off, x, jnp.round(x))


def round_invtanh_fwd(x, scale, off=False, alpha_scale=1.):
  return round_invtanh(x, scale, off=off), (x, scale, alpha_scale)


def round_invtanh_bwd(res, g):
  (x, scale, alpha_scale) = res

  # parameter to scale the softness/steepness.
  alpha = 1.9

  inv_tanh_coeff = (1 + scale * jnp.sign(g) * .5 / jnp.arctanh(
      alpha / 2) * jnp.arctanh(
      (x - jnp.round(x)) * alpha))
  ewgs_coeff = (1 + scale * jnp.sign(g) * (x - jnp.round(x)))
  return (g * (inv_tanh_coeff * alpha_scale + ewgs_coeff * (1 - alpha_scale)
               ), None, None, None)


round_invtanh.defvjp(round_invtanh_fwd, round_invtanh_bwd)

# Type 2: Gradients pushing towards quantization state.


# psgd https://arxiv.org/abs/2005.11035 (like)
@jax.custom_vjp
def round_psgd(x, scale, off=False):
  return jnp.where(off, x, jnp.round(x))


def round_psgd_fwd(x, scale, off=False):
  return round_psgd(x, scale, off=off), (x, scale)


def round_psgd_bwd(res, g):
  (x, scale) = res

  rel_shift = .0  # 0. -.25 -.5
  abs_shift = .0  # -1.

  return (g * (1 + scale * (jnp.abs((x - jnp.round(x))) + rel_shift
                            ) + abs_shift), None, None)


round_psgd.defvjp(round_psgd_fwd, round_psgd_bwd)


@jax.custom_vjp
def round_fsig(x, scale, off=False):
  return jnp.where(off, x, jnp.round(x))


def round_fsig_fwd(x, scale, off=False):
  return round_fsig(x, scale, off=off), (x, scale)


def round_fsig_bwd(res, g):
  (x, scale) = res

  # Fast sigmoid derivative
  def fsig_deriv(x):
    return 1 / (1 + jnp.abs(x))**2

  # 2 is a parameter to scale the softness/steepness.
  return (g * (1 + scale * jnp.sign(g) * (fsig_deriv((x + .5 - jnp.round(
      x + .5)) * 2.))), None, None)


round_fsig.defvjp(round_fsig_fwd, round_fsig_bwd)

# https://arxiv.org/abs/2103.12593
# Copied from https://github.com/byin-cwi/Efficient-spiking-networks/\
# blob/main/DVS128/srnn_class_scnn_enc.ipynb


@jax.custom_vjp
def round_gaussian(x, scale, off=False):
  return jnp.where(off, x, jnp.round(x))


def round_gaussian_fwd(x, scale, off=False):
  return round_gaussian(x, scale, off=off), (x, scale)


def round_gaussian_bwd(res, g):
  (x, scale) = res

  lens = .5

  def gaussian_deriv(x):
    return jnp.exp(-(x**2) / (2 * lens**2)) / jnp.sqrt(2 * jnp.pi) / lens

  return (g * (1 + scale * jnp.sign(g) * gaussian_deriv((x + .5 - jnp.round(
      x + .5)) * 3)), None, None)


round_gaussian.defvjp(round_gaussian_fwd, round_gaussian_bwd)

# https://arxiv.org/abs/2103.12593
# Copied from https://github.com/byin-cwi/Efficient-spiking-networks/\
# blob/main/DVS128/srnn_class_scnn_enc.ipynb


@jax.custom_vjp
def round_multi_gaussian(x, scale, off=False):
  return jnp.where(off, x, jnp.round(x))


def round_multi_gaussian_fwd(x, scale, off=False):
  return round_multi_gaussian(x, scale, off=off), (x, scale)


def round_multi_gaussian_bwd(res, g):
  (x, scale) = res

  # Fast sigmoid derivative
  lens = .5
  hight = .15
  scale_gaussian = 6.0

  def gaussian_fn(x, mu, sigma):
    return jnp.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / jnp.sqrt(
        2 * jnp.pi) / sigma

  def multi_gaussian_deriv(x):
    return gaussian_fn(x, mu=0., sigma=lens) * (
        1. + hight) - gaussian_fn(
        x, mu=lens, sigma=scale_gaussian * lens) * hight - gaussian_fn(
        x, mu=- lens, sigma=scale_gaussian * lens) * hight

  return (g * (1 + scale * jnp.sign(g) * multi_gaussian_deriv((
      x + .5 - jnp.round(x + .5)) * 3)), None, None)


round_multi_gaussian.defvjp(round_multi_gaussian_fwd, round_multi_gaussian_bwd)


#
# Calibration functions
#


def max_init(x, bits, sign, axis=None):
  return jnp.where(jnp.max(x) == 0, 1 / 2**bits, jnp.max(jnp.abs(x),
                                                         axis=axis))


# def double_mean_init(x, bits, sign):
#   return jnp.where(jnp.max(x) == 0, 1 / 2**bits, 2 * jnp.mean(jnp.abs(x)))


def gaussian_init(x, bits, sign, axis=None):
  mu = jnp.mean(x, axis=axis)
  sigma = jnp.std(x, axis=axis)
  return jnp.where(jnp.max(x) == 0, 1 / 2**bits, jnp.maximum(jnp.abs(
      mu - 3 * sigma), jnp.abs(mu + 3 * sigma)))


def percentile_init(x, bits, sign, perc, axis=None):
  return jnp.where(jnp.max(x) == 0, 1 / 2**bits,
                   jnp.percentile(jnp.abs(x), perc, axis=axis))


#
# Quantizer
#


class uniform_static(nn.Module):
  bits: int = 8
  act: bool = False
  round_fn: Callable = round_psgd
  init_fn: Callable = max_init
  g_scale: float = 0.
  maxabs_w: float = None

  @nn.compact
  def __call__(self, x: Array, sign: bool = True) -> Array:
    if type(self.bits) == int:
      assert (
          self.bits > 1
      ), "Bit widths below 2 bits are not supported but got bits: "\
          + str(self.bits)

    if sign:
      num_levels = 2 ** (self.bits - 1) - 1
    else:
      num_levels = 2 ** (self.bits) - 1

    xmax = self.variable(
        'quant_params', 'dynamic_range_no_train', jnp.ones, (1,))
    if self.is_mutable_collection('quant_params'):
      xmax.value = self.init_fn(x, bits=self.bits, sign=sign)
      xmax.value = jnp.where(xmax.value == 0, 1., xmax.value)

    # clip x
    if sign:
      x = x / xmax.value
      x = jnp.clip(x, -1., 1.) * xmax.value
    else:
      x = x / xmax.value
      x = jnp.clip(x, 0., 1.) * xmax.value

    scale = xmax.value / num_levels
    return self.round_fn(x / scale, self.g_scale) * scale


class parametric_d(nn.Module):
  bits: int = 8
  act: bool = False
  round_fn: Callable = round_psgd
  init_fn: Callable = max_init
  g_scale: float = 0.
  clip_quant_grads: bool = True
  maxabs_w: float = None

  # parametric homogenouse quantization
  # Based on LEARNED STEP SIZE QUANTIZATION
  # https://arxiv.org/abs/1902.08153.
  @nn.compact
  def __call__(self, inputs: Array, sign: bool = True) -> Array:

    v = inputs

    if sign:
      q_pos = 2 ** (self.bits - 1) - 1
      q_neg = -q_pos

    else:
      q_pos = 2 ** (self.bits) - 1
      q_neg = 0

    if self.act:
      n_wf = v.shape[1:]
    else:
      n_wf = v.shape

    # Intialize step size. Step size only changes when init is called or apply
    # with mutable = ['quant_params'].
    step_size = self.variable('quant_params', 'step_size', jnp.ones, (1,))
    if self.is_mutable_collection('quant_params'):
      step_size.value = jnp.ones((1,))
      step_size.value *= self.init_fn(inputs,
                                      bits=self.bits,
                                      sign=sign) / jnp.sqrt(q_pos)

    gradScaleFactor = 1 / jnp.sqrt(q_pos * np.prod(n_wf) + 1e-6)
    # print('step_size = ' + str(step_size.value))
    # print('scale = '+str(gradScaleFactor))

    @jax.custom_vjp
    def gradscale(x, scale, d):
      return x

    def gradscale_fwd(x, scale, d):
      return gradscale(x, scale, d), (scale, d)

    def gradscale_bwd(res, g):
      (scale, d) = res
      # clip gradient
      if d is not None:
        return jnp.clip(g * scale, a_min=-d, a_max=d), None, None
      else:
        return g * scale, None, None
    gradscale.defvjp(gradscale_fwd, gradscale_bwd)

    s = gradscale(step_size.value, gradScaleFactor,
                  step_size.value if self.clip_quant_grads else None)
    v = v / s
    v = jnp.clip(v, q_neg, q_pos)
    vbar = self.round_fn(v, self.g_scale)
    return vbar * s


class DuQ(nn.Module):
  bits: int = 4
  act: bool = False  # not used, possibly for different init for acts
  g_scale: float = 0.
  round_fn: Callable = round_ste
  maxabs_w: float = None

  # Differentiable and unified Quantization (DuQ)
  # Based on PROFIT.
  # https://arxiv.org/pdf/2008.04693.pdf
  @nn.compact
  def __call__(self, inputs: Array, sign: bool = True) -> Array:

    @jax.custom_vjp
    def DuQ_round_quant(x, n_lvl):
      return self.round_fn(x * (n_lvl - 1), self.g_scale) / (n_lvl - 1)

    def DuQ_round_quant_fwd(x, n_lvl):
      return DuQ_round_quant(x, n_lvl), (None,)

    def DuQ_round_quant_bwd(res, g):
      return g, None

    DuQ_round_quant.defvjp(DuQ_round_quant_fwd, DuQ_round_quant_bwd)

    if self.bits == -1:  # option to have a pass through quantizer
      return inputs

    x = inputs

    if sign:
      n_lv = 2 ** (self.bits - 1)
    else:
      n_lv = 2 ** self.bits

    a = self.param('a', constant(-1), (1,))
    c = self.param('c', constant(-1), (1,))

    x = jax.nn.hard_tanh(x / a)
    x = DuQ_round_quant(x, n_lv) * c

    return jnp.where(a == -1, inputs, jnp.array(x, dtype=inputs.dtype))


class prune(nn.Module):

  @nn.compact
  def __call__(self, inputs: Array, sign: bool = True) -> Array:

    @jax.custom_vjp
    def grad_zero(x):
      return x

    def grad_zero_fwd(x):
      return grad_zero(x), (None,)

    def grad_zero_bwd(res, g):
      return g * 0.,

    grad_zero.defvjp(grad_zero_fwd, grad_zero_bwd)

    mask = self.param('mask', constant(1), inputs.shape)

    return jnp.array(inputs * grad_zero(mask), dtype=inputs.dtype)


class parametric_d_xmax(nn.Module):
  bits: int = 4  # here its just init bits
  act: bool = False
  xmax_min: float = 2**-8
  xmax_max: float = 127
  d_min: float = 2**-12
  d_max: float = 1
  round_fn: Callable = round_ste
  init_fn: Callable = None
  g_scale: float = 0.
  ceil_tolerance: float = 0.0
  maxabs_w: float = None
  bitwidth_min: int = 2

  # Parametric heterogenous quantization.
  # Based on MIXED PRECISION DNNS.
  # https://openreview.net/pdf?id=Hyx0slrFvH
  @nn.compact
  def __call__(self, inputs: Array, sign: bool = True) -> Array:

    x = inputs

    def quantize_pow2(v):
      # return 2 ** round_psgd(jnp.log2(v), 0)
      return 2 ** round_psgd(jnp.log2(v), 0)

    @jax.custom_vjp
    def ceilpass(x):
      return jnp.ceil(x - self.ceil_tolerance)

    def ceilpass_fwd(x):
      return ceilpass(x), (None,)

    def ceilpass_bwd(res, g):
      return (g,)

    ceilpass.defvjp(ceilpass_fwd, ceilpass_bwd)

    if sign:
      num_levels = 2 ** (self.bits - 1) - 1
    else:
      num_levels = 2 ** self.bits - 1

    xmax_max = self.variable('quant_config', 'max_xmax',  # noqa: F841
                             lambda x: float(self.xmax_max), (1,))
    xmax_min = self.variable('quant_config', 'min_xmax',  # noqa: F841
                             lambda x: float(self.xmax_min), (1,))
    d_max = self.variable('quant_config', 'max_d',  # noqa: F841
                          lambda x: float(self.d_max), (1,))
    d_min = self.variable('quant_config', 'min_d',  # noqa: F841
                          lambda x: float(self.d_min), (1,))

    # Intialize step size. Step size only changes when init is called or apply
    # with mutable = ['quant_params'].
    d = self.variable('quant_params', 'step_size', jnp.ones, (1,))
    xmax = self.variable(
        'quant_params', 'dynamic_range', jnp.ones, (1,))

    act_mb = self.variable('act_size', 'act_mb', jnp.ones, (1,))
    weight_mb = self.variable('weight_size', 'weight_mb', jnp.ones, (1,))
    bw = self.bits
    if self.is_mutable_collection('quant_params'):

      if self.init_fn is None:
        # Original init from MixedDNN paper.
        if self.act:
          xmax.value = 2**-3 * (2. ** bw - 1)
          d.value = 2**-3
        else:
          maxabs_w = self.maxabs_w if self.maxabs_w is not None else jnp.max(
              jnp.abs(inputs))
          if bw > 4:
            d.value = 2**(jnp.ceil(jnp.log2(maxabs_w / (2**(bw - 1) - 1))))
          else:
            d.value = 2**(jnp.floor(jnp.log2(maxabs_w / (2**(bw - 1) - 1))))
          xmax.value = d.value * (2 ** (bw - 1) - 1)
      else:
        # Improved init with custom function.
        xmax.value = self.init_fn(inputs, bits=self.bits, sign=sign)
        xmax.value = jnp.where(xmax.value == 0, 1., xmax.value)
        d.value = xmax.value / num_levels

    # Ensure that stepsize is in specified range (and a power of two).
    d = jnp.clip(d.value, self.d_min, self.d_max)
    # d = quantize_pow2(d)
    # Ensure that dynamic range is in specified range.
    xmax = jnp.clip(xmax.value, self.xmax_min, self.xmax_max)

    # Aux scope to compute network size on the fly.
    real_xmax = round_psgd(xmax / d, 0) * d  # for size computation
    if self.is_mutable_collection('act_size'):

      if self.act:
        n_wf = inputs.shape[1:]
        if sign:
          act_mb.value = np.prod(
              n_wf) * jnp.mean(jnp.maximum((ceilpass(jnp.log2(
                  (real_xmax / d) + 1)
              ) + 1), self.bitwidth_min))
        else:
          act_mb.value = np.prod(
              n_wf) * jnp.mean(jnp.maximum((ceilpass(jnp.log2(
                  (real_xmax / d) + 1))
              ), self.bitwidth_min))
      else:
        act_mb.value = 0.

    if self.is_mutable_collection('weight_size'):
      if self.act:
        weight_mb.value = 0.
      else:
        n_wf = inputs.shape
        if sign:
          weight_mb.value = np.prod(
              n_wf) * jnp.mean(jnp.maximum((ceilpass(jnp.log2(
                  (real_xmax / d) + 1)
              ) + 1), self.bitwidth_min))
        else:
          weight_mb.value = np.prod(
              n_wf) * jnp.mean(jnp.maximum((ceilpass(jnp.log2(
                  (real_xmax / d) + 1))
              ), self.bitwidth_min))

    # clip x
    if sign:
      x = x / xmax
      x = jnp.clip(x, -1., 1.) * xmax
    else:
      x = x / xmax
      x = jnp.clip(x, 0., 1.) * xmax

    return d * self.round_fn(x / d, self.g_scale)
