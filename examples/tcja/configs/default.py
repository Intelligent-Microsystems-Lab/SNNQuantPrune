# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Originally copied from https://github.com/google/flax/tree/main/examples

"""Default Hyperparameter configuration."""


# lambda x, y, z: cross_entropy_loss(x/config.num_frames, y, z)

import ml_collections
from functools import partial
from spiking_learning import multi_step_LIF, atan
from train_utils import mse_loss
import jax.numpy as jnp


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.seed = 203853699

  # As defined in the `models` module.
  config.model = "CextNet"

  # neural dynamics
  config.neuron_dynamics = partial(multi_step_LIF, spike_fn=atan, tau=2.0)

  # data set
  config.dataset = "dvs_gesture_tfds"
  config.tfds_data_dir = "gs://imagenet_clemens/tensorflow_datasets"
  config.num_frames = 20
  config.split_by = "number"
  config.num_classes = 11

  # questionable parameters
  config.resolution_scale = 1.0
  config.channels = 128

  # optimizer and training
  config.loss_fn = partial(mse_loss, T=1)

  config.optimizer = "adam"

  config.learning_rate = 0.001
  config.warmup_epochs = 5
  config.num_epochs = 1024
  config.weight_decay = 0.0  # 1e-4
  config.dropout = 0.5
  config.batch_size = 16
  config.eval_batch_size = 16
  config.smoothing = 0.0

  config.pretrained = None  # '../data/tcja_test/checkpoint_max.pth'

  # If num_train_steps==-1 then the number of training steps is calculated from
  # num_epochs using the entire dataset. Similarly for steps_per_eval.
  config.log_every_steps = 2
  config.num_train_steps = -1
  config.steps_per_eval = -1
  config.cache = True
  config.num_devices = 8
  config.dtype = jnp.float32

  return config
