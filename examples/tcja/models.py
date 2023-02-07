# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Based on https://github.com/ridgerchu/TCJA/blob/main/src/dvs128.py

"""Flax implementation of "TCJA-SNN: Temporal-Channel Joint Attention for
Spiking Neural Networks"
https://arxiv.org/pdf/2206.10177.pdf"""

import sys
import ml_collections
from functools import partial
from typing import Any, Callable

from flax import linen as nn
import jax.numpy as jnp
import jax

sys.path.append('tcja')
from tcja_load_pretrained_weights import tcja_load_pretrained_weights  # noqa: E402, E501

sys.path.append("..")
from spiking_learning import SpikingBlock  # noqa: E402

sys.path.append("..")
from flax_qconv import QuantConv  # noqa: E402
from flax_qdense import QuantDense  # noqa: E402

Array = jnp.ndarray


class CextNet(nn.Module):
  num_classes: int = 11
  dtype: Any = jnp.float32
  config: dict = ml_collections.FrozenConfigDict({})
  load_model_fn: Callable = tcja_load_pretrained_weights

  @nn.compact
  def __call__(self, inputs: Array, trgt: Array, train: bool, rng: Any,
               u_state=None, online=False) -> Array:

    def TCJA(x_seq, kernel_size=4, config=None, i=0):
      x = jnp.moveaxis(jnp.mean(x_seq, axis=[2, 3]), (0, 1, 2), (1, 0, 2))
      x_c = jnp.moveaxis(x, (0, 1, 2), (0, 2, 1))

      sparse_nums = jnp.sum(jnp.reshape(
          x_c, x_c.shape[:1] + (-1,)) != 0., axis=-1) / jnp.prod(jnp.array(
              x_c.shape[1:]))
      self.sow('intermediates', 'conv_tcja1_'
               + str(i) + '_inpt_min', jnp.max(sparse_nums))
      self.sow('intermediates', 'conv_tcja1_' + str(i)
               + '_inpt_mean', jnp.mean(sparse_nums))
      conv_t_out = QuantConv(features=x_seq.shape[0],
                             kernel_size=[kernel_size],
                             padding='SAME',
                             use_bias=False,
                             dtype=x_seq.dtype,
                             config=config.quant,
                             bits=config.quant.bits,
                             g_scale=config.quant.g_scale,)(x_c)
      sparse_nums = jnp.sum(jnp.reshape(
          conv_t_out, conv_t_out.shape[:1] + (-1,)) != 0., axis=-1) / jnp.prod(
          jnp.array(conv_t_out.shape[1:]))
      self.sow('intermediates', 'conv_tcja1_'
               + str(i) + '_out_min', jnp.max(sparse_nums))
      self.sow('intermediates', 'conv_tcja1_' + str(i)
               + '_out_mean', jnp.mean(sparse_nums))

      conv_t_out = jnp.moveaxis(conv_t_out, (0, 1, 2), (1, 2, 0))

      sparse_nums = jnp.sum(jnp.reshape(
          x, x.shape[:1] + (-1,)) != 0., axis=-1) / jnp.prod(
          jnp.array(x.shape[1:]))
      self.sow('intermediates', 'conv_tcja2_'
               + str(i) + '_inpt_min', jnp.max(sparse_nums))
      self.sow('intermediates', 'conv_tcja2_' + str(i)
               + '_inpt_mean', jnp.mean(sparse_nums))
      conv_c_out = QuantConv(features=x_seq.shape[-1],
                             kernel_size=[kernel_size],
                             padding='SAME',
                             use_bias=False,
                             dtype=x_seq.dtype,
                             config=config.quant,
                             bits=config.quant.bits,
                             g_scale=config.quant.g_scale,)(x)
      sparse_nums = jnp.sum(jnp.reshape(
          conv_c_out, conv_c_out.shape[:1] + (-1,)) != 0., axis=-1) / jnp.prod(
          jnp.array(conv_c_out.shape[1:]))
      self.sow('intermediates', 'conv_tcja2_'
               + str(i) + '_out_min', jnp.max(sparse_nums))
      self.sow('intermediates', 'conv_tcja2_' + str(i)
               + '_out_mean', jnp.mean(sparse_nums))

      conv_c_out = jnp.moveaxis(conv_c_out, (0, 1, 2), (1, 0, 2))

      out = jax.nn.sigmoid(conv_c_out * conv_t_out)

      y_seq = x_seq * out[:, :, None, None, :]

      return y_seq

    norm = partial(nn.BatchNorm,
                   use_running_average=not train,
                   momentum=0.9,
                   epsilon=1e-5,
                   use_bias=True,
                   use_scale=True,
                   dtype=self.dtype)

    x = jnp.swapaxes(inputs, 0, 1)

    for i in range(3):
      layer = SpikingBlock(
          connection_fn=QuantConv(features=self.config.channels,
                                  kernel_size=(3, 3),
                                  padding=((1, 1), (1, 1)),
                                  use_bias=False,
                                  dtype=self.dtype,
                                  config=self.config.quant,
                                  bits=self.config.quant.bits,
                                  g_scale=self.config.quant.g_scale,
                                  ),
          neural_dynamics=self.config.neuron_dynamics(dtype=self.dtype),
          norm_fn=norm(),
      )
      carry = layer.initialize_carry(
          x, layer.connection_fn, layer.norm_fn, dtype=self.dtype)

      sparse_nums = jnp.sum(jnp.reshape(
          x, x.shape[:2] + (-1,)) != 0., axis=-1) / jnp.prod(
          jnp.array(x.shape[2:]))
      self.sow('intermediates', 'conv_' + str(i)
               + '_inpt_min', jnp.max(sparse_nums))
      self.sow('intermediates', 'conv_' + str(i)
               + '_inpt_mean', jnp.mean(sparse_nums))
      _, x = layer(carry, x)
      sparse_nums = jnp.sum(jnp.reshape(
          x, x.shape[:2] + (-1,)) != 0., axis=-1) / jnp.prod(
          jnp.array(x.shape[2:]))
      self.sow('intermediates', 'conv_' + str(i)
               + '_out_min', jnp.max(sparse_nums))
      self.sow('intermediates', 'conv_' + str(i)
               + '_out_mean', jnp.mean(sparse_nums))

      # max pooling
      x = jax.lax.reduce_window(x, -jnp.inf, jax.lax.max,
                                (1, 1, 2, 2, 1), (1, 1, 2, 2, 1),
                                [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)])

    for i in range(2):
      layer = SpikingBlock(
          connection_fn=QuantConv(features=self.config.channels,
                                  kernel_size=(3, 3),
                                  padding=((1, 1), (1, 1)),
                                  use_bias=False,
                                  dtype=self.dtype,
                                  config=self.config.quant,
                                  bits=self.config.quant.bits,
                                  g_scale=self.config.quant.g_scale,
                                  ),
          neural_dynamics=self.config.neuron_dynamics(dtype=self.dtype),
          norm_fn=norm(),
      )
      carry = layer.initialize_carry(
          x, layer.connection_fn, layer.norm_fn, dtype=self.dtype)

      sparse_nums = jnp.sum(jnp.reshape(
          x, x.shape[:2] + (-1,)) != 0., axis=-1) / jnp.prod(
          jnp.array(x.shape[2:]))
      self.sow('intermediates', 'conv_t_' + str(i)
               + '_inpt_min', jnp.max(sparse_nums))
      self.sow('intermediates', 'conv_t_' + str(i)
               + '_inpt_mean', jnp.mean(sparse_nums))
      _, x = layer(carry, x)
      sparse_nums = jnp.sum(jnp.reshape(
          x, x.shape[:2] + (-1,)) != 0., axis=-1) / jnp.prod(
          jnp.array(x.shape[2:]))
      self.sow('intermediates', 'conv_t_' + str(i)
               + '_out_min', jnp.max(sparse_nums))
      self.sow('intermediates', 'conv_t_' + str(i)
               + '_out_mean', jnp.mean(sparse_nums))

      x = TCJA(x, config=self.config, i=i)

      # max pooling
      x = jax.lax.reduce_window(x, -jnp.inf, jax.lax.max,
                                (1, 1, 2, 2, 1), (1, 1, 2, 2, 1),
                                [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)])

    x = jnp.transpose(x, (0, 1, 4, 2, 3))  # for pytorch compatability
    x = jnp.reshape(x, x.shape[:2] + (-1,))

    if train:
      # dropout
      rng, prng = jax.random.split(rng, 2)
      mask = jax.random.bernoulli(
          prng, p=self.config.dropout, shape=x.shape)
      # that is potentially different
      x = x * mask  # jnp.repeat(jnp.expand_dims(mask, 0), x.shape[0], axis=0)

    layer = SpikingBlock(
        connection_fn=QuantDense(
            self.config.channels * 2 * 2, use_bias=False, dtype=self.dtype,
            config=self.config.quant,
            bits=self.config.quant.bits,
            g_scale=self.config.quant.g_scale,
        ),
        neural_dynamics=self.config.neuron_dynamics(dtype=self.dtype),
    )
    carry = layer.initialize_carry(x, layer.connection_fn, dtype=self.dtype)

    sparse_nums = jnp.sum(jnp.reshape(
        x, x.shape[:2] + (-1,)) != 0., axis=-1) / jnp.prod(
        jnp.array(x.shape[2:]))
    self.sow('intermediates', 'dense1_inpt_min', jnp.max(sparse_nums))
    self.sow('intermediates', 'dense1_inpt_mean', jnp.mean(sparse_nums))
    _, x = layer(carry, x)
    sparse_nums = jnp.sum(jnp.reshape(
        x, x.shape[:2] + (-1,)) != 0., axis=-1) / jnp.prod(
        jnp.array(x.shape[2:]))
    self.sow('intermediates', 'dense1_out_min', jnp.max(sparse_nums))
    self.sow('intermediates', 'dense1_out_mean', jnp.mean(sparse_nums))

    if train:
      # dropout
      rng, prng = jax.random.split(rng, 2)
      mask = jax.random.bernoulli(
          prng, p=self.config.dropout, shape=x.shape)  # [1:])
      # that is potentially different
      x = x * mask  # jnp.repeat(jnp.expand_dims(mask, 0), x.shape[0], axis=0)

    layer = SpikingBlock(
        connection_fn=QuantDense(self.num_classes * 10,
                                 use_bias=False, dtype=self.dtype,
                                 config=self.config.quant,
                                 bits=self.config.quant.bits,
                                 g_scale=self.config.quant.g_scale,),
        neural_dynamics=self.config.neuron_dynamics(dtype=self.dtype),
    )
    carry = layer.initialize_carry(x, layer.connection_fn, dtype=self.dtype)

    sparse_nums = jnp.sum(jnp.reshape(
        x, x.shape[:2] + (-1,)) != 0., axis=-1) / jnp.prod(
        jnp.array(x.shape[2:]))
    self.sow('intermediates', 'dense2_inpt_min', jnp.max(sparse_nums))
    self.sow('intermediates', 'dense2_inpt_mean', jnp.mean(sparse_nums))
    _, x = layer(carry, x)
    sparse_nums = jnp.sum(jnp.reshape(
        x, x.shape[:2] + (-1,)) != 0., axis=-1) / jnp.prod(
        jnp.array(x.shape[2:]))
    self.sow('intermediates', 'dense2_out_min', jnp.max(sparse_nums))
    self.sow('intermediates', 'dense2_out_mean', jnp.mean(sparse_nums))

    # "vote"
    x = jnp.mean(x, 0)
    x = jnp.mean(x.reshape(x.shape[:1] + (-1, 10)), axis=-1)

    return x, None
