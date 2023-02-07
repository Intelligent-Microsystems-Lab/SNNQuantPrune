# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Unit Test for QuantDense


from absl.testing import absltest
from absl.testing import parameterized

from flax import linen as nn
from flax import optim
from flax.core import freeze
from jax import random
from jax.nn import initializers
import jax
import jax.numpy as jnp
from flax import jax_utils
import functools

import ml_collections

from flax_qdense import QuantDense


class DQG(nn.Module):
  """A simple fully connected model with QuantDense"""

  @nn.compact
  def __call__(self, x, channels, config, rng):
    """Description of CNN forward pass
    Args:
        x: an array (inputs)
        channels: an array containing number of channels for each layer
        config: bit width for gradient in the backward pass
    Returns:
        An array containing the result.
    """
    rng, subkey = jax.random.split(rng, 2)
    x = QuantDense(
        features=channels[0],
        kernel_init=initializers.lecun_normal(),
        config=config,
        use_bias=False,
    )(x, subkey)
    rng, subkey = jax.random.split(rng, 2)
    x = QuantDense(
        features=channels[1],
        kernel_init=initializers.lecun_normal(),
        config=config,
        use_bias=False,
    )(x, subkey)
    return x


class Dlinen(nn.Module):
  """Same model as above but with nn.Dense"""

  @nn.compact
  def __call__(self, x, channels):
    """Description of CNN forward pass
    Args:
        x: an array (inputs)
        channels: an array containing number of channels for each layer
    Returns:
        An array containing the result.
    """
    x = nn.Dense(
        features=channels[0],
        kernel_init=initializers.lecun_normal(),
        use_bias=False,
    )(x)
    x = nn.Dense(
        features=channels[1],
        kernel_init=initializers.lecun_normal(),
        use_bias=False,
    )(x)
    return x


def create_optimizer(params, learning_rate):
  optimizer_def = optim.GradientDescent(learning_rate=learning_rate)
  optimizer = optimizer_def.create(params)
  return optimizer


def cross_entropy_loss(logits, labels):
  return -jnp.mean(jnp.sum(labels * logits, axis=-1))


# Train step for QuantDense layer
def train_step_dense_quant_grad(optimizer, batch, out_channels, config, rng):
  """Train for a single step."""

  def loss_fn(params):
    logits = DQG().apply(
        {"params": params}, batch["image"], out_channels, config, rng
    )
    loss = cross_entropy_loss(logits, batch["label"])
    return loss, logits

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grad = grad_fn(optimizer.target)
  optimizer = optimizer.apply_gradient(grad)
  return optimizer


# Train step for nn.Dense layer
def train_step_dense(optimizer, batch, out_channels):
  """Train for a single step."""

  def loss_fn(params):
    logits = Dlinen().apply(
        {"params": params}, batch["image"], out_channels
    )
    loss = cross_entropy_loss(logits, batch["label"])
    return loss, logits

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grad = grad_fn(optimizer.target)
  optimizer = optimizer.apply_gradient(grad)
  return optimizer


# Test data for QuantDense
def dense_test_data():
  return (
      dict(
          testcase_name="base_case",
          examples=512,
          inp_channels=100,
          channels=[20, 10],
          config=ml_collections.FrozenConfigDict({}),
          numerical_tolerance=1e-8,
      ),
      dict(
          testcase_name="base_case_1024examples_1channel",
          examples=1024,
          inp_channels=1,
          channels=[1, 1],
          config=ml_collections.FrozenConfigDict({}),
          numerical_tolerance=1e-8,
      ),
      dict(
          testcase_name="base_case_200channels",
          examples=256,
          inp_channels=1,
          channels=[200, 1],
          config=ml_collections.FrozenConfigDict({}),
          numerical_tolerance=1e-7,
      ),
  )


class QuantDenseTest(parameterized.TestCase):
  @parameterized.named_parameters(*dense_test_data())
  def test_QuantDense_vs_nnDense(
      self, examples, inp_channels, channels, config, numerical_tolerance
  ):
    """
    Unit test to check whether QuantDense does exactly the same as
    nn.Dense when gradient quantization is turned off.
    """

    # create initial data
    key = random.PRNGKey(8627169)
    key, subkey1, subkey2 = random.split(key, 3)
    data_x = random.uniform(
        subkey1,
        (jax.device_count(), examples, inp_channels),
        minval=-1,
        maxval=1,
    )
    data_y = random.uniform(
        subkey2,
        (jax.device_count(), examples, channels[1]),
        minval=-1,
        maxval=1,
    )

    # setup QuantDense
    key, subkey1, subkey2, subkey3 = random.split(key, 4)
    params_qgrad = DQG().init(
        subkey1, jnp.take(data_x, 0, axis=0), channels, config, subkey2
    )["params"]
    optimizer_quant_grad = create_optimizer(params_qgrad, 1)

    # setup nn.Dense with parameters from QuantDense
    params = freeze(
        {
            "Dense_0": params_qgrad["QuantDense_0"],
            "Dense_1": params_qgrad["QuantDense_1"],
        }
    )
    optimizer = create_optimizer(params, 1)

    # check that weights are initially equal for both layers
    assert (
        params_qgrad["QuantDense_0"]["kernel"] == params["Dense_0"]["kernel"]
    ).all() and (
        params_qgrad["QuantDense_1"]["kernel"] == params["Dense_1"]["kernel"]
    ).all(), "Initial parameters not equal"

    optimizer_quant_grad = jax_utils.replicate(optimizer_quant_grad)
    optimizer = jax_utils.replicate(optimizer)

    p_train_step_dense_quant_grad = jax.pmap(
        functools.partial(
            train_step_dense_quant_grad,
            out_channels=channels,
            config=config,
            rng=subkey3,
        ),
        axis_name="batch",
    )
    p_train_step_dense = jax.pmap(
        functools.partial(
            train_step_dense,
            out_channels=channels,
        ),
        axis_name="batch",
    )

    # one backward pass
    optimizer_quant_grad = p_train_step_dense_quant_grad(
        optimizer_quant_grad,
        {"image": data_x, "label": data_y},
    )
    optimizer = p_train_step_dense(
        optimizer,
        {"image": data_x, "label": data_y},
    )

    # determine difference between nn.Dense and QuantDense
    dense1_diff = optimizer.target["Dense_1"]["kernel"] - (
        optimizer_quant_grad.target["QuantDense_1"]["kernel"])
    dense0_diff = optimizer.target["Dense_0"]["kernel"] - (
        optimizer_quant_grad.target["QuantDense_0"]["kernel"])
    self.assertLessEqual(
        jnp.mean(
            (
                jnp.mean(
                    abs(dense1_diff)
                ) / jnp.mean(abs(optimizer.target["Dense_1"]["kernel"]))
            ) + (
                jnp.mean(
                    abs(dense0_diff)
                ) / jnp.mean(abs(optimizer.target["Dense_0"]["kernel"]))
            )
        ),
        numerical_tolerance,
    )


if __name__ == "__main__":
  absltest.main()
