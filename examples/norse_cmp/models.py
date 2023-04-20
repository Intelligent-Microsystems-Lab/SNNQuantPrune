# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer

import sys
import ml_collections
from functools import partial
from typing import Any

from flax import linen as nn
import jax.numpy as jnp

sys.path.append("../..")
from spiking_learning import SpikingBlock  # noqa: E402

Array = jnp.ndarray


class NorseMNISTNet(nn.Module):
  num_classes: int = 10
  dtype: Any = jnp.float32
  config: dict = ml_collections.FrozenConfigDict({})

  @nn.compact
  def __call__(self, inputs: Array, trgt: Array, train: bool, rng: Any,
               u_state=None, online=False) -> Array:

    norm = partial(nn.BatchNorm,
                   use_running_average=not train,
                   momentum=0.9,
                   epsilon=1e-5,
                   use_bias=True,
                   use_scale=True,
                   dtype=self.dtype)

    inputs = jnp.mean(inputs, axis=-1, keepdims=True)
    x = jnp.swapaxes(inputs, 0, 1)

    x = jnp.reshape(x, (x.shape[0], x.shape[1], -1))

    layer = SpikingBlock(
        connection_fn=nn.Dense(
            100, use_bias=False, dtype=self.dtype,
        ),
        neural_dynamics=self.config.neuron_dynamics(dtype=self.dtype),
        norm_fn=norm(),
    )
    carry = layer.initialize_carry(
        x, layer.connection_fn, layer.norm_fn, dtype=self.dtype)
    _, x = layer(carry, x)

    layer = SpikingBlock(
        connection_fn=nn.Dense(self.num_classes,
                               use_bias=False, dtype=self.dtype,),
        neural_dynamics=self.config.neuron_dynamics(dtype=self.dtype),
    )
    carry = layer.initialize_carry(x, layer.connection_fn, dtype=self.dtype)
    _, x = layer(carry, x)

    x = jnp.mean(x, 0)

    return x, None
