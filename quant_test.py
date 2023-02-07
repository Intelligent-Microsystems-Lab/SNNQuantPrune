# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Unit Test for quant


from absl.testing import absltest
from absl.testing import parameterized
from flax.core import freeze, unfreeze

from jax import random
import jax
import jax.numpy as jnp


import numpy as np
import re


from quant import (
    uniform_static,
    parametric_d,
    parametric_d_xmax,
)

jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_disable_most_optimizations', True)
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_disable_jit", True)


def signed_uniform_max_scale_quant_ste_equality_data():
  return (
      dict(
          x_dim=100,
          y_dim=30,
          dtype=jnp.int8,
      ),
      dict(
          x_dim=100,
          y_dim=30,
          dtype=jnp.int16,
      ),
  )


def signed_uniform_max_scale_quant_ste_unique_data():
  return (
      dict(
          x_dim=100,
          y_dim=30,
          bits=3,
          scale=23,
      ),
      dict(
          x_dim=100,
          y_dim=30,
          bits=4,
          scale=39,
      ),
      dict(
          x_dim=100,
          y_dim=30,
          bits=5,
          scale=0.57,
      ),
      dict(
          x_dim=100,
          y_dim=30,
          bits=6,
          scale=4319,
      ),
      dict(
          x_dim=100,
          y_dim=30,
          bits=7,
          scale=0.835,
      ),
      dict(
          x_dim=100,
          y_dim=30,
          bits=8,
          scale=3,
      ),
      dict(
          x_dim=100,
          y_dim=30,
          bits=9,
          scale=780,
      ),
      dict(
          x_dim=7100,
          y_dim=530,
          bits=10,
          scale=0.38,
      ),
      dict(
          x_dim=9101,
          y_dim=261,
          bits=11,
          scale=654,
      ),
      dict(
          x_dim=3300,
          y_dim=632,
          bits=12,
          scale=153,
      ),
  )


def signed_uniform_max_scale_quant_ste_unique_data_ext():
  return (
      dict(
          x_dim=1030,
          y_dim=300,
          bits=2,
          scale=23,
      ),
      dict(
          x_dim=1200,
          y_dim=930,
          bits=13,
          scale=813,
      ),
      dict(
          x_dim=7100,
          y_dim=930,
          bits=14,
          scale=4897,
      ),
      dict(
          x_dim=2700,
          y_dim=930,
          bits=15,
          scale=561,
      ),
  )


class QuantOpsTest(parameterized.TestCase):
  @parameterized.product(
      signed_uniform_max_scale_quant_ste_equality_data(),
      quantizer=(uniform_static,
                 parametric_d, parametric_d_xmax)
  )
  def test_equality_native_dtypes(
      self, x_dim, y_dim, dtype, quantizer,
  ):
    key = random.PRNGKey(8627169)

    key, subkey = jax.random.split(key)
    data = jax.random.randint(
        subkey,
        (x_dim, y_dim),
        minval=np.iinfo(dtype).min,
        maxval=np.iinfo(dtype).max,
    )
    data = data.at[0, 0].set(np.iinfo(dtype).min)
    data = jnp.clip(
        data, a_min=np.iinfo(dtype).min + 1, a_max=np.iinfo(dtype).max
    )
    data = jnp.array(data, jnp.float64)

    bits = int(re.split("(\d+)", dtype.__name__)[1])  # noqa: W605

    key, subkey = jax.random.split(key)
    if quantizer.__name__ == 'parametric_d_xmax':
      variables = quantizer(
          bits, xmax_max=np.iinfo(dtype).max).init(subkey, data)
    else:
      variables = quantizer(bits).init(subkey, data)

    scale = 1
    if 'quant_params' in variables:
      if 'dynamic_range' not in variables['quant_params']:
        if 'step_size' in variables['quant_params']:
          scale = variables['quant_params']['step_size']

    if quantizer.__name__ == 'parametric_d_xmax':
      dataq = quantizer(bits, xmax_max=np.iinfo(
          dtype).max).apply(variables, data * scale)
    else:
      dataq = quantizer(bits).apply(variables, data * scale)

    np.testing.assert_allclose(data, dataq / scale)

  @parameterized.product(
      signed_uniform_max_scale_quant_ste_unique_data(
      ) + signed_uniform_max_scale_quant_ste_unique_data_ext(),
      quantizer=(parametric_d_xmax, uniform_static,
                 parametric_d),
  )
  def test_unique_values(
      self, x_dim, y_dim, bits, scale, quantizer
  ):
    def quantize_pow2(v):
      return 2 ** jnp.round(jnp.log2(v), 0)

    key = random.PRNGKey(8627169)

    key, subkey = jax.random.split(key)
    data = (
        jax.random.uniform(subkey, (x_dim, y_dim), minval=-1, maxval=1) * scale
    )
    data = data.at[0, 0].set(scale)

    key, subkey = jax.random.split(key)
    if quantizer.__name__ == 'parametric_d_xmax':
      variables = quantizer(bits, xmax_max=2**32, d_max=2**32,
                            d_min=-1, ).init(subkey, data)
    else:
      variables = quantizer(bits).init(subkey, data)

    if 'quant_params' in variables:
      if quantizer.__name__ == 'parametric_d_xmax':
        variables = unfreeze(variables)

        if bits > 4:
          variables['quant_params']['step_size'] = 2**(
              jnp.ceil(jnp.log2(scale / (2**(bits - 1) - 1))))
        else:
          variables['quant_params']['step_size'] = 2**(
              jnp.floor(jnp.log2(scale / (2**(bits - 1) - 1))))
        variables['quant_params']['dynamic_range'] = scale
        real_xmax = jnp.round(
            scale / variables['quant_params']['step_size'], 0
        ) * variables['quant_params']['step_size']
        variables = freeze(variables)

        data = data / scale * real_xmax
      if quantizer.__name__ == 'parametric_d':
        variables = unfreeze(variables)
        variables['quant_params']['step_size'] = scale / (2 ** (bits - 1) - 1)
        variables = freeze(variables)

    if quantizer.__name__ == 'parametric_d_xmax':
      dataq = quantizer(bits, xmax_max=2 ** 32, d_max=2**32,
                        d_min=-1, ).apply(variables, data)
    else:
      dataq = quantizer(bits).apply(variables, data)

    if quantizer.__name__ == 'parametric_d_xmax':
      self.assertEqual(
          len(np.unique(dataq)), real_xmax / variables['quant_params'
                                                       ]['step_size'] * 2 + 1
      )
    else:
      self.assertEqual(
          len(np.unique(dataq)), (2 ** (bits) - 1)
      )

  # @parameterized.product(signed_uniform_max_scale_quant_ste_unique_data())
  # def test_parametric_d(self, x_dim, y_dim, bits, scale):

  #   #
  #   # DISCLAIMER: too big and and too small of a bit-width breaks unit tests.
  #   #

  #   rng = random.PRNGKey(8627169)

  #   rng, init_rng, data_rng = jax.random.split(rng, 3)
  #   data = (
  #       jax.random.uniform(data_rng, (x_dim, y_dim), minval=-1, maxval=1
  #                          ) * scale
  #   )

  #   quant_fn = parametric_d(bits=bits)

  #   def loss_fn(x, params):
  #     logits = quant_fn.apply(params, x)
  #     return jnp.sum(logits)

  #   params = quant_fn.init(init_rng, data)
  #   grad_fn = jax.grad(loss_fn, argnums=1)

  #   num_levels = 2 ** (bits - 1) - 1
  #   grad_scale = 1 / jnp.sqrt(num_levels * np.prod(data.shape) + 1e-6)
  #   params_step_size = params['quant_params']['step_size']

  #   # all outside upper
  #   g = grad_fn(jnp.abs(data) + num_levels * params_step_size, params)
  #   np.testing.assert_allclose(g['quant_params']['step_size'] / (
  #       num_levels * grad_scale), x_dim * y_dim, atol=6.e-05, rtol=7e-6)

  #   # all inside on point
  #   g = grad_fn(jnp.ones_like(data) * params_step_size, params)
  #   # numerical tol.
  #   np.testing.assert_allclose(
  #       g['quant_params']['step_size'], 5e-5, atol=6.e-05)

  #   # all inside full off point
  #   g = grad_fn(jnp.ones_like(data) * params_step_size * .5, params)
  #   np.testing.assert_allclose(jnp.abs(g['quant_params']['step_size'] / (
  #       x_dim * y_dim)), .5 * grad_scale, atol=6.e-05)

  #   # all outside lower
  #   g = grad_fn(-jnp.abs(data) - num_levels * params_step_size, params)
  #   np.testing.assert_allclose(g['quant_params']['step_size'] / (
  #       num_levels * grad_scale), -x_dim * y_dim, atol=6.e-05, rtol=7e-6)

  @parameterized.product(signed_uniform_max_scale_quant_ste_unique_data(
  ) + signed_uniform_max_scale_quant_ste_unique_data_ext())
  def test_grad_d_xmax(self, x_dim, y_dim, bits, scale):

    #
    # DISCLAIMER: Too big and and too small of a scale break unit tests.
    # Because clipping has an effect on gradients!
    # (e.g. set xmax and d limits correctly)
    #

    rng = random.PRNGKey(8627169)

    rng, init_rng, data_rng = jax.random.split(rng, 3)
    data = (
        jax.random.uniform(data_rng, (x_dim, y_dim), minval=-1, maxval=1
                           ) * scale
    )
    data = data.at[0, 0].set(scale - 0.001)

    quant_fn = parametric_d_xmax(
        bits=bits, xmax_max=2**16, d_min=1e-12, d_max=scale + 1)

    def loss_fn(x, params):
      logits = quant_fn.apply(params, x)
      return jnp.sum(logits) - jax.lax.stop_gradient(0.1 * jnp.sum(logits))

    params = quant_fn.init(init_rng, data)
    grad_fn = jax.grad(loss_fn, argnums=1)
    grad_fn_err = jax.grad(loss_fn, argnums=0)

    params_step_size = params['quant_params']['step_size']
    params_dynamic_range = params['quant_params']['dynamic_range']

    if params_dynamic_range > scale:
      pass
    else:
      data = data / scale * params_dynamic_range

    # Grads for Err

    # All inside.
    g = grad_fn_err(data, params)
    self.assertEqual(jnp.round(jnp.sum(g)), np.prod(data.shape))

    if params_dynamic_range > scale:
      scale = params_dynamic_range

    # 10% outside upper
    g = grad_fn_err(data + .1 * scale, params)

    self.assertEqual(jnp.sum(g), jnp.sum(
        data + .1 * scale < params_dynamic_range))

    # 10% outside lower
    g = grad_fn_err(data - .1 * scale, params)
    self.assertEqual(jnp.sum(g), jnp.sum(
        data - .1 * scale > -params_dynamic_range))

    # Dynamic Range.

    # All inside.
    g = grad_fn(data, params)
    self.assertLessEqual(jnp.abs(g['quant_params']['dynamic_range']), 1e-2)

    # 10% outside upper
    g = grad_fn(data + .1 * scale, params)
    self.assertLessEqual(jnp.abs(g['quant_params']['dynamic_range'] - jnp.sum(
        data + .1 * scale > params_dynamic_range)), 0.07)

    # 10% outside lower
    g = grad_fn(data - .1 * scale, params)
    self.assertLessEqual(jnp.abs(g['quant_params']['dynamic_range'] - (
        -jnp.sum(data - .1 * scale < -params_dynamic_range))), 0.002)

    # Step Size.

    # all outside upper
    g = grad_fn(jnp.ones_like(data) + params_dynamic_range, params)
    self.assertEqual(g['quant_params']['step_size'], 0)

    # all outside lower
    g = grad_fn(-jnp.ones_like(data) - params_dynamic_range, params)
    self.assertEqual(g['quant_params']['step_size'], 0)

    # all inside on point
    g = grad_fn(jnp.ones_like(data) * params_step_size, params)
    # numerical tol.
    self.assertLessEqual(g['quant_params']['step_size'], 0.)

    # all inside full off point
    g = grad_fn(jnp.ones_like(data) * params_step_size * .5, params)
    np.testing.assert_allclose(
        g['quant_params']['step_size'] / np.prod(data.shape), -.5, atol=1e-7)


if __name__ == "__main__":
  absltest.main()
