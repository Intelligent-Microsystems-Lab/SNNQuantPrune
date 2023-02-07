# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer

import functools


from typing import Any, Callable, Sequence

import jax
from jax import dtypes
from jax import random
from flax import linen as nn
import jax.numpy as jnp

import numpy as np

from jax._src.nn.initializers import lecun_normal


Array = jnp.ndarray
DType = Any


def uniform(scale=1e-2, dtype: DType = jnp.float_) -> Callable:
  """Builds an initializer that returns real uniformly-distributed random
   arrays.

Args:
  scale: optional; the upper and lower bound of the random distribution.
  dtype: optional; the initializer's default dtype.

Returns:
  An initializer that returns arrays whose values are uniformly distributed
  in the range ``[-scale, scale)``.

"""

  def init(key, shape, dtype=dtype):
    dtype = dtypes.canonicalize_dtype(dtype)
    return random.uniform(key, shape, dtype) * scale * 2 - scale

  return init


def static_init(val=1.0, dtype: DType = jnp.float_) -> Callable:
  def init(key, shape, dtype=dtype):
    dtype = dtypes.canonicalize_dtype(dtype)
    return jnp.ones(shape, dtype) * val

  return init


def normal_shift(
    bias=0, scale=1e-2, no_sign_flip=True, dtype: DType = jnp.float_
) -> Callable:
  """Builds an initializer that returns real uniformly-distributed random
   arrays.

Args:
  scale: optional; the upper and lower bound of the random distribution.
  dtype: optional; the initializer's default dtype.

Returns:
  An initializer that returns arrays whose values are uniformly distributed
  in the range ``[-scale, scale)``.

"""

  def init(key, shape, dtype=dtype):
    dtype = dtypes.canonicalize_dtype(dtype)

    x = random.normal(key, shape, dtype) * scale + bias
    if no_sign_flip:
      x = jnp.abs(x)
    return x

  return init


@jax.custom_vjp
def debug(x):
  return x


def debug_fwd(x):
  return debug(x), x


def debug_bwd(res, g):
  import pdb

  pdb.set_trace()

  return (g,)


debug.defvjp(debug_fwd, debug_bwd)


class gsis(nn.Module):
  sigmoid_bias: float = 2
  sigmoid_scale: float = 2
  theta: float = 0.1
  fn: Callable = lambda x: 1 / (1 + (2 * jnp.pi / 2 * x) ** 2)

  @nn.compact
  def __call__(self, x):
    @jax.custom_vjp
    def gsis_fn(x):
      return x

    def gsis_fn_fwd(x):
      return gsis_fn(x), x

    def gsis_fn_bwd(res, g):
      x = res

      # alpha = 2  # fixed at two !!!!
      # scale = self.theta * jax.nn.relu(1 - (jnp.abs(x) * 2)) # piecewise
      scale = 1 + self.theta * self.fn(x)  # atan
      # scale = (1 + self.theta * jnp.abs(x - (x >= .5)))

      return (g * scale,)

    gsis_fn.defvjp(gsis_fn_fwd, gsis_fn_bwd)

    alpha = self.param(
        "upscale",
        normal_shift(self.sigmoid_bias, self.sigmoid_scale),
        (x.shape[-1],),
    )

    # pre process
    x = jax.nn.sigmoid(x * alpha)

    return gsis_fn(x)


@jax.custom_vjp
def fast_sigmoid(x):
  # if not dtype float grad ops wont work
  return jnp.array(x >= 0.0, dtype=x.dtype)


def fast_sigmoid_fwd(x):
  return fast_sigmoid(x), x


def fast_sigmoid_bwd(res, g):
  x = res
  alpha = 10

  scale = 1 / (alpha * jnp.abs(x) + 1.0) ** 2
  return (g * scale,)


fast_sigmoid.defvjp(fast_sigmoid_fwd, fast_sigmoid_bwd)


@jax.custom_vjp
def slayer(x):
  # if not dtype float grad ops wont work
  return jnp.array(x >= 0.0, dtype=x.dtype)


def slayer_fwd(x):
  return slayer(x), x


def slayer_bwd(res, g):
  x = res

  scale = jnp.exp(-jnp.abs(x) * 5)
  return (g * scale,)


slayer.defvjp(slayer_fwd, slayer_bwd)


@jax.custom_vjp
def smooth_step(x):
  # if not dtype float grad ops wont work
  return jnp.array(x >= 0.0, dtype=x.dtype)


def smooth_step_fwd(x):
  return smooth_step(x), x


def smooth_step_bwd(res, g):
  x = res

  scale = jnp.logical_and((x < 0.5), (x >= -0.5))
  return (g * scale,)


smooth_step.defvjp(smooth_step_fwd, smooth_step_bwd)


@jax.custom_vjp
def piecewise_linear(x):
  # if not dtype float grad ops wont work
  return jnp.array(x >= 0.0, dtype=x.dtype)


def piecewise_linear_fwd(x):
  return piecewise_linear(x), x


def piecewise_linear_bwd(res, g):
  x = res

  # mask = jnp.logical_and((x > . 5), (x <= -.5))
  scale = jax.nn.relu(1 - (jnp.abs(x) * 2))
  return (g * scale,)  # * mask,


piecewise_linear.defvjp(piecewise_linear_fwd, piecewise_linear_bwd)


@jax.custom_vjp
def atan(x):
  # if not dtype float grad ops wont work
  return jnp.array(x >= 0.0, dtype=x.dtype)


def atan_fwd(x):
  return atan(x), x


def atan_bwd(res, g):
  # originally from SpikingJelly

  x = res
  alpha = 2

  shared_c = g / (1 + (alpha * jnp.pi / 2 * x) ** 2)
  return (alpha / 2 * shared_c,)


atan.defvjp(atan_fwd, atan_bwd)


class leaky_current_based_IF_rel_refactory(nn.Module):
  beta: float
  alpha: float
  alpharp: float
  spike_fn: Callable
  connection_fn: Callable
  wrp: float = 1.0

  """
  From "Synaptic Plasticity Dynamics for Deep Continuous Local Learning
  (DECOLLE)" - https://arxiv.org/abs/1811.10766
  """

  @nn.compact
  def __call__(self, carry, s_in):

    sQ, sP, sR, sS = carry

    Q = self.beta * sQ + (1 - self.beta) * s_in
    P = self.alpha * sP + (1 - self.alpha) * sQ
    R = self.alpharp * sR - (1 - self.alpharp) * sS * self.wrp
    U = self.connection_fn(P) + R
    S = self.spike_fn(U)

    return (Q, P, R, S), U

  @staticmethod
  def initialize_carry(inputs, connection_fn):
    x = connection_fn(inputs)
    return (
        jnp.zeros_like(inputs, dtype=jnp.float32),
        jnp.zeros_like(inputs, dtype=jnp.float32),
        jnp.zeros(x.shape, dtype=jnp.float32),
        jnp.zeros(x.shape, dtype=jnp.float32),
    )


class DecolleSpikingBlock(nn.Module):
  connection_fn: Callable
  loss_type: Callable
  num_classes: int
  neural_dynamics: Callable
  pool_window: Sequence[int] = (1, 1)
  train: bool = True
  drop_out: float = 0.5

  """
  From "Synaptic Plasticity Dynamics for Deep Continuous Local Learning
  (DECOLLE)" - https://arxiv.org/abs/1811.10766
  """

  @functools.partial(
      nn.transforms.scan,
      variable_broadcast="params",
      split_rngs={"params": False, "dropout": True},
  )
  @nn.compact
  def __call__(self, carry, pair):
    inputs, trgt = pair

    carry, u = self.neural_dynamics(connection_fn=self.connection_fn)(
        carry, inputs
    )
    u_p = nn.max_pool(u, self.pool_window, strides=self.pool_window)
    s_ = fast_sigmoid(u_p)

    # local learning
    flatten_size = np.prod(u_p.shape[1:])
    w_ro = self.param(
        "w_ro", lecun_normal(), (self.num_classes, flatten_size)
    )
    stdv = 0.5 / np.sqrt(self.num_classes)  # lc_ampl
    b_ro = self.param("b_ro", uniform(stdv), (self.num_classes,))

    @jax.custom_vjp
    def decolle(x, w, b, trgt):
      out_local = jnp.dot(x, w.transpose()) + b

      return out_local

    def decolle_fwd(x, w, b, trgt):
      out_local = decolle(x, w, b, trgt)

      return out_local, (out_local, w, trgt, x.shape)

    def decolle_bwd(res, g):
      (out_local, w, trgt, shape) = res

      err = jax.grad(
          lambda x: jnp.mean(jnp.mean(self.loss_type(x, trgt)))
      )(out_local)
      grad = jnp.dot(err, w)

      return grad, jnp.zeros_like(w), jnp.zeros((err.shape[-1])), None

    decolle.defvjp(decolle_fwd, decolle_bwd)

    sd_ = nn.Dropout(self.drop_out)(s_, deterministic=not self.train)
    # reshape has to be compatible with decolle pytorch
    sd_ = jnp.reshape(
        jnp.moveaxis(sd_, (0, 1, 2, 3), (0, 2, 3, 1)), (sd_.shape[0], -1)
    )
    out_local = decolle(sd_, w_ro, b_ro, trgt)

    return carry, (s_, out_local)

  @staticmethod
  def initialize_carry(inputs, connection_fn, neural_dynamics):
    return neural_dynamics(connection_fn=connection_fn).initialize_carry(
        inputs[0, :], connection_fn
    )


class parametric_leaky_IF(nn.Module):
  init_tau: float
  spike_fn: Callable
  v_threshold: float = 1.0
  v_reset: float = 0.0
  pre_spike_fn: Callable = None
  dtype: Any = jnp.float32

  """
  From "Incorporating Learnable Membrane Time Constant to Enhance Learning of
  Spiking Neural Networks" - https://arxiv.org/pdf/2007.05785.pdf
  """

  @nn.compact
  def __call__(self, u, s_in):

    tau = self.param(
        "tau",
        static_init(-jnp.log(self.init_tau - 1), dtype=self.dtype),
        (1,),
    )
    v_threshold = jnp.array([self.v_threshold], dtype=self.dtype)
    v_reset = jnp.array([self.v_reset], dtype=self.dtype)

    u += (s_in - (u - v_reset)) * jax.nn.sigmoid(tau)

    s = self.spike_fn(u - v_threshold)

    u = jnp.where(s, v_reset, u)

    return u, s


class multi_step_LIF(nn.Module):
  tau: float
  spike_fn: Callable
  v_threshold: float = 1.0
  v_reset: float = 0.0
  pre_spike_fn: Callable = None
  dtype: Any = jnp.float32

  """
  From "TCJA-SNN: Temporal-Channel Joint Attention for Spiking Neural Networks"
  - https://arxiv.org/pdf/2206.10177.pdf
  """

  @nn.compact
  def __call__(self, u, s_in):

    tau = jnp.array([self.tau], dtype=self.dtype)
    v_threshold = jnp.array([self.v_threshold], dtype=self.dtype)
    v_reset = jnp.array([self.v_reset], dtype=self.dtype)

    u += (s_in - (u - v_reset)) / tau

    s = self.spike_fn(u - v_threshold)

    u = jnp.where(s, v_reset, u)

    return u, s


class LIF(nn.Module):
  init_tau: float
  spike_fn: Callable
  v_threshold: float = 1.0
  v_reset: float = 0.0
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, u, s_in):
    tau = self.param("tau", uniform(self.init_tau), (u.shape[-1],))
    v_threshold = jnp.array([self.v_threshold], dtype=self.dtype)
    v_reset = jnp.array([self.v_reset], dtype=self.dtype)

    u = u * jax.nn.sigmoid(tau) + s_in

    s = self.spike_fn(u - v_threshold)

    u = jnp.where(s > 0.5, v_reset, u)

    return u, s


class SpikingBlock(nn.Module):
  connection_fn: Callable
  neural_dynamics: Callable
  norm_fn: Callable = None

  @nn.remat
  @functools.partial(
      nn.transforms.scan,
      variable_broadcast="params",
      variable_carry="batch_stats",
      split_rngs={"params": False},
  )
  @nn.compact
  def __call__(self, u, inputs):
    x = self.connection_fn(inputs)

    if self.norm_fn:
      x = self.norm_fn(x)

    u, s = self.neural_dynamics(u, x)

    return u, s

  @staticmethod
  def initialize_carry(
      inputs, connection_fn, norm_fn=None, dtype=jnp.float32
  ):
    x = connection_fn(inputs[0, :])
    if norm_fn:
      x = norm_fn(x)

    return jnp.zeros_like(x, dtype=dtype)
