# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Originally copied from https://github.com/google/flax/tree/main/examples

"""Default Hyperparameter configuration."""

import ml_collections
from functools import partial
from spiking_learning import multi_step_LIF, atan
from train_utils import cross_entropy_loss
import jax.numpy as jnp


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.seed = 203853699

  # As defined in the `models` module.
  config.model = "NorseMNISTNet"

  # neural dynamics
  config.neuron_dynamics = partial(multi_step_LIF, spike_fn=atan, tau=2.0)

  # data set
  config.dataset = "mnist"
  config.tfds_data_dir = None
  config.num_frames = 32
  config.pretrained = None
  config.num_classes = 10

  # optimizer and training
  config.loss_fn = cross_entropy_loss

  config.optimizer = "adam"

  config.learning_rate = 0.002
  config.warmup_epochs = 0
  config.num_epochs = 5
  config.weight_decay = 0.0  # 1e-4
  # config.dropout = 0.5
  config.batch_size = 256
  config.eval_batch_size = 16
  config.smoothing = 0.0

  config.quant = ml_collections.ConfigDict()

  # If num_train_steps==-1 then the number of training steps is calculated from
  # num_epochs using the entire dataset. Similarly for steps_per_eval.
  config.log_every_steps = 235
  config.num_train_steps = -1
  config.steps_per_eval = -1
  config.cache = True
  config.num_devices = 4
  config.dtype = jnp.bfloat16

  return config
