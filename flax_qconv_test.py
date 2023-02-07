# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Unit Test for QuantConv
# 1. Unite test to compare nn.Conv layer with QuantConv. Simply training a one
# layer convolutional network initialized with all zeros and comparing the
# weight after update


from absl.testing import absltest
from absl.testing import parameterized

import functools
from flax import linen as nn
from flax import optim
from flax.core import freeze
from jax import random
from jax.nn import initializers
import jax
from jax import lax
import jax.numpy as jnp
from flax import jax_utils

from flax_qconv import QuantConv


class CQG(nn.Module):
  """A simple fully connected model with QuantConv and config = None for
  no quantization"""

  @nn.compact
  def __call__(
      self, x, features, kernel_size, strides, padding, config, rng
  ):
    rng1, rng2 = jax.random.split(rng, 2)
    x = QuantConv(
        features=features[0],
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        kernel_init=initializers.lecun_normal(),
        config=config,
        use_bias=False,
    )(x, rng1)
    x = QuantConv(
        features=features[1],
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        kernel_init=initializers.lecun_normal(),
        config=config,
        use_bias=False,
    )(x, rng2)
    return x


class Clinen(nn.Module):
  """Same model as above but with nn.Conv"""

  @nn.compact
  def __call__(self, x, features, kernel_size, strides, padding):
    x = nn.Conv(
        features=features[0],
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        kernel_init=initializers.lecun_normal(),
        use_bias=False,
    )(x)
    x = nn.Conv(
        features=features[1],
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
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


# Train step for QuantConv layer
def train_step_conv_quant_grad(
    optimizer,
    batch,
    features,
    kernel_size,
    strides,
    padding,
    config,
    rng,
):
  """Train for a single step."""

  def loss_fn(params):
    logits = CQG().apply(
        {"params": params},
        batch["image"],
        features,
        kernel_size,
        strides,
        padding,
        config,
        rng,
    )
    loss = cross_entropy_loss(logits, batch["label"])
    return loss, logits

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grad = grad_fn(optimizer.target)
  grad = lax.pmean(grad, axis_name="batch")
  optimizer = optimizer.apply_gradient(grad)
  return logits, optimizer


# Train step for nn.Conv layer
def train_step_conv(optimizer, batch, features, kernel_size, strides, padding):
  """Train for a single step."""

  def loss_fn(params):
    logits = Clinen().apply(
        {"params": params},
        batch["image"],
        features,
        kernel_size,
        strides,
        padding,
    )
    loss = cross_entropy_loss(logits, batch["label"])
    return loss, logits

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grad = grad_fn(optimizer.target)
  grad = lax.pmean(grad, axis_name="batch")
  optimizer = optimizer.apply_gradient(grad)
  return logits, optimizer


# Test data for QuantConv
def conv_test_data():
  return (
      dict(
          testcase_name="base_case",
          examples=128,
          dimX=28,
          dimY=28,
          dimX_out=28,
          dimY_out=28,
          inp_channels=1,
          features=(10, 20),
          kernel_size=(2, 2),
          strides=(1, 1),
          padding="SAME",
          config={},
          numerical_tolerance=0.0,
      ),
      dict(
          testcase_name="base_case_1x1_input",
          examples=128,
          dimX=1,
          dimY=1,
          dimX_out=1,
          dimY_out=1,
          inp_channels=1,
          features=(10, 20),
          kernel_size=(2, 2),
          strides=(1, 1),
          padding="SAME",
          config={},
          numerical_tolerance=0.0,
      ),
      dict(
          testcase_name="base_case_13x17_input",
          examples=128,
          dimX=13,
          dimY=17,
          dimX_out=13,
          dimY_out=17,
          inp_channels=1,
          features=(10, 20),
          kernel_size=(2, 2),
          strides=(1, 1),
          padding="SAME",
          config={},
          numerical_tolerance=0.0,
      ),
      dict(
          testcase_name="base_case_1x1_kernel",
          examples=128,
          dimX=28,
          dimY=28,
          dimX_out=28,
          dimY_out=28,
          inp_channels=1,
          features=(10, 20),
          kernel_size=(1, 1),
          strides=(1, 1),
          padding="SAME",
          config={},
          numerical_tolerance=0.0,
      ),
      dict(
          testcase_name="base_case_3x7_kernel",
          examples=128,
          dimX=28,
          dimY=28,
          dimX_out=28,
          dimY_out=28,
          inp_channels=1,
          features=(10, 20),
          kernel_size=(3, 7),
          strides=(1, 1),
          padding="SAME",
          config={},
          numerical_tolerance=0.0,
      ),
      dict(
          testcase_name="base_case_valid_padding",
          examples=128,
          dimX=28,
          dimY=28,
          dimX_out=26,
          dimY_out=26,
          inp_channels=1,
          features=(10, 20),
          kernel_size=(2, 2),
          strides=(1, 1),
          padding="VALID",
          config={},
          numerical_tolerance=0.0,
      ),
      dict(
          testcase_name="base_case_3715_padding",
          examples=128,
          dimX=28,
          dimY=28,
          dimX_out=46,
          dimY_out=38,
          inp_channels=1,
          features=(10, 20),
          kernel_size=(2, 2),
          strides=(1, 1),
          padding=((3, 7), (1, 5)),
          config={},
          numerical_tolerance=0.0,
      ),
      dict(
          testcase_name="base_case_2_2_stride",
          examples=128,
          dimX=28,
          dimY=28,
          dimX_out=7,
          dimY_out=7,
          inp_channels=1,
          features=(10, 20),
          kernel_size=(2, 2),
          strides=(2, 2),
          padding="SAME",
          config={},
          numerical_tolerance=0.0,
      ),
      dict(
          testcase_name="base_case_3_7_stride",
          examples=128,
          dimX=28,
          dimY=28,
          dimX_out=4,
          dimY_out=1,
          inp_channels=1,
          features=(10, 20),
          kernel_size=(2, 2),
          strides=(3, 7),
          padding="SAME",
          config={},
          numerical_tolerance=0.0,
      ),
  )


class QuantConvTest(parameterized.TestCase):
  @parameterized.named_parameters(*conv_test_data())
  def test_QuantConv_vs_nnConv(
      self,
      examples,
      dimX,
      dimY,
      dimX_out,
      dimY_out,
      inp_channels,
      features,
      kernel_size,
      strides,
      padding,
      config,
      numerical_tolerance,
  ):
    """
    Unit test to check whether QuantConv does exactly the same as
    nn.Conv when gradient quantization is turned off.
    """
    # create initial data
    key = random.PRNGKey(8627169)
    key, subkey1, subkey2 = random.split(key, 3)
    data_x = (
        random.uniform(
            subkey1,
            (jax.device_count(), examples, dimX, dimY, inp_channels),
            minval=-1,
            maxval=1,
            dtype=jnp.float32,
        ) * 100
    )
    data_y = (
        random.uniform(
            subkey2,
            (
                jax.device_count(),
                examples,
                dimX_out,
                dimY_out,
                features[-1],
            ),
            minval=-1,
            maxval=1,
            dtype=jnp.float32,
        ) * 100
    )
    # setup QuantConv
    key, subkey1, subkey2, subkey3 = random.split(key, 4)
    params_quant_grad = CQG().init(
        subkey1,
        jnp.take(data_x, 0, axis=0),
        features,
        kernel_size,
        strides,
        padding,
        config,
        subkey2,
    )["params"]
    optimizer_qgrad = create_optimizer(params_quant_grad, 1)

    p_train_step_conv_quant_grad = jax.pmap(
        functools.partial(
            train_step_conv_quant_grad,
            features=features,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            config=config,
            rng=subkey3,
        ),
        axis_name="batch",
    )
    p_train_step_conv = jax.pmap(
        functools.partial(
            train_step_conv,
            features=features,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
        ),
        axis_name="batch",
    )

    # setup nn.Conv
    params = freeze(
        {
            "Conv_0": params_quant_grad["QuantConv_0"],
            "Conv_1": params_quant_grad["QuantConv_1"],
        }
    )
    optimizer = create_optimizer(params, 1)

    # check that weights are initially equal
    assert (
        optimizer_qgrad.target["QuantConv_0"] == optimizer.target["Conv_0"]
    ) and (
        optimizer_qgrad.target["QuantConv_1"] == optimizer.target["Conv_1"]
    ), "Initial parameters not equal"

    optimizer_qgrad = jax_utils.replicate(optimizer_qgrad)
    optimizer = jax_utils.replicate(optimizer)

    # one backward pass
    logits_quant, optimizer_qgrad = p_train_step_conv_quant_grad(
        optimizer_qgrad,
        {"image": data_x, "label": data_y},
    )
    logits, optimizer = p_train_step_conv(
        optimizer,
        {"image": data_x, "label": data_y},
    )
    # determine difference between nn.Conv and QuantConv
    diff_conv0 = optimizer.target["Conv_0"]["kernel"] - (
        optimizer_qgrad.target["QuantConv_0"]["kernel"])
    diff_conv1 = optimizer.target["Conv_1"]["kernel"] - (
        optimizer_qgrad.target["QuantConv_1"]["kernel"])
    self.assertLessEqual(
        (
            (
                jnp.mean(
                    abs(diff_conv0)
                )
            ) / jnp.mean(abs(optimizer.target["Conv_0"]["kernel"])) + (
                jnp.mean(abs(diff_conv1))
            ) / jnp.mean(abs(optimizer.target["Conv_1"]["kernel"]))
        ) / 2,
        numerical_tolerance,
    )


if __name__ == "__main__":
  absltest.main()
