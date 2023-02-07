# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Originally copied from https://github.com/google/flax/tree/main/examples

"""Training script.
The data is loaded using tensorflow_datasets.
"""


import functools
import subprocess
# import time

from absl import app
from absl import flags
from absl import logging

from clu import platform
# from clu import metric_writers
# from clu import periodic_actions

from flax import jax_utils
from flax.training import common_utils

import jax
# import optax
from jax import random

import tensorflow as tf

import numpy as np
import jax.numpy as jnp

import models

from torch.utils.data import DataLoader

import ml_collections
from ml_collections import config_flags

from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

from train_utils import (
    TrainState,
    #     restore_checkpoint,
    #     save_checkpoint,
    #     create_learning_rate_fn,
    create_model,
    create_train_state,
    #     train_step,
    #     eval_step,
    #     sync_batch_stats,
)


# from jax.config import config
# config.update("jax_debug_nans", True)
# config.update("jax_disable_jit", True)
# config.update('jax_platform_name', 'cpu')


FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Directory to store model data.')
config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True)


def train_and_evaluate(config: ml_collections.ConfigDict,
                       workdir: str) -> TrainState:
  """Execute model training and evaluation loop.
  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.
  Returns:
    Final TrainState.
  """

  logging.info('Git commit: ' + subprocess.check_output(
      ['git', 'rev-parse', 'HEAD']).decode('ascii').strip())
  logging.info(config)

  rng = random.PRNGKey(config.seed)

  dev_num = config.num_devices if 'num_devices' in config else\
      jax.local_device_count()

  train_set = DVS128Gesture('/home/clemens/datasets/DVS128Gesture', train=True,
                            data_type='frame', split_by='number',
                            frames_number=config.num_frames)
  train_iter = DataLoader(
      dataset=train_set,
      batch_size=config.batch_size,
      shuffle=True,
      num_workers=16,
      drop_last=True,
      pin_memory=True)

  def check_quant_obj(x):
    if not hasattr(x, 'keys'):
      return True
    if 'kernel' in x.keys():
      return True
    return False

  def sparsity_compute(x):
    if not hasattr(x, 'keys'):
      return None
    else:

      # prune
      array = jnp.multiply(x['kernel'], x['prune_0']['mask'])

      # quant
      n_lvl = 2 ** (config.quant.bits - 1)
      array = jax.nn.hard_tanh(array / x['DuQ_0']['a'])
      array = (jnp.round(array * (n_lvl - 1)) / (n_lvl - 1)) * x['DuQ_0']['a']

      return np.sum(array != 0.) / np.prod(array.shape)

  model_dtype = config.dtype if 'dtype' in config else jnp.float32

  model_cls = getattr(models, config.model)

  model = create_model(
      model_cls=model_cls, num_classes=config.num_classes,
      model_dtype=model_dtype, config=config)

  image_size = 128
  rng, subkey = jax.random.split(rng, 2)
  state = create_train_state(
      subkey, config, model, image_size, lambda x: x)
  state = model.load_model_fn(state, workdir)

  layer_sparse = jax.tree_map(
      sparsity_compute, state.params['params'], is_leaf=check_quant_obj)

  track_intermediates = []

  peval = jax.pmap(functools.partial(model.apply, trgt=None, train=False,
                   online=False, mutable=['intermediates', 'batch_stats']))

  state = jax_utils.replicate(state, devices=jax.devices(
  )[:config.num_devices] if 'num_devices' in config else jax.devices())

  step = 0
  for frame, label in train_iter:
    if (step % 15) == 0:
      logging.info('eval step: ' + str(step))
    rng_list = jax.random.split(rng, dev_num + 1)
    rng = rng_list[0]

    batch = {'dvs_matrix': jnp.reshape(jnp.transpose(frame.numpy(),
                                                     (0, 1, 3, 4, 2)),
                                       (config.num_devices, -1,
                                        config.num_frames, image_size,
                                        image_size, 2)), 'label':
             jnp.reshape(label.numpy(), (config.num_devices, -1))}

    test = peval({'params': state.params['params'],
                  'batch_stats': state.batch_stats},
                 batch['dvs_matrix'], rng=rng_list[1:],
                 rngs={'dropout': rng_list[1:]},)
    track_intermediates.append(test[1]['intermediates'])
    step += 1

  acc_inter = common_utils.stack_forest(track_intermediates)

  file = ['name,weights,inputs,outputs,T,C,M,P,Q,R,S,HS,WS\n',
          'Conv1,'
          + str(float(layer_sparse['QuantConv_0']))
          + ',' + str(np.mean(acc_inter['conv_0_inpt_mean'][0])) + ',' + str(
            np.mean(acc_inter['conv_0_out_mean'][0]))
          + ',20,2,128,128,128,3,3,1,1\n',
          'Conv2,'
          + str(float(layer_sparse['QuantConv_1']))
          + ',' + str(np.mean(acc_inter['conv_1_inpt_mean'][0])) + ',' + str(
            np.mean(acc_inter['conv_1_out_mean'][0]))
          + ',20,128,128,64,64,3,3,1,1\n',
          'Conv3,'
          + str(float(layer_sparse['QuantConv_2']))
          + ',' + str(np.mean(acc_inter['conv_2_inpt_mean'][0])) + ',' + str(
            np.mean(acc_inter['conv_2_out_mean'][0]))
          + ',20,128,128,32,32,3,3,1,1\n',
          'Conv4,'
          + str(float(layer_sparse['QuantConv_3']))
          + ',' + str(np.mean(acc_inter['conv_t_0_inpt_mean'][0])) + ',' + str(
            np.mean(acc_inter['conv_t_0_out_mean'][0]))
          + ',20,128,128,16,16,3,3,1,1\n',
          'TCJA11,'
          + str(float(layer_sparse['QuantConv_4']))
          + ',' + str(np.mean(acc_inter['conv_tcja1_0_inpt_mean'][0])) + ','
          + str(
            np.mean(acc_inter['conv_tcja1_0_out_mean'][0]))
          + ',20,20,20,1,256,1,4,1,1\n',
          'TCJA12,'
          + str(float(layer_sparse['QuantConv_5']))
          + ',' + str(np.mean(acc_inter['conv_tcja2_0_inpt_mean'][0])) + ','
          + str(
            np.mean(acc_inter['conv_tcja2_0_out_mean'][0]))
          + ',128,128,20,1,256,1,4,1,1\n',
          'Conv5,'
          + str(float(layer_sparse['QuantConv_6']))
          + ',' + str(np.mean(acc_inter['conv_t_1_inpt_mean'][0])) + ','
          + str(
            np.mean(acc_inter['conv_t_1_out_mean'][0]))
          + ',20,128,128,8,8,3,3,1,1\n',
          'TCJA21,'
          + str(float(layer_sparse['QuantConv_7']))
          + ',' + str(np.mean(acc_inter['conv_tcja1_1_inpt_mean'][0])) + ','
          + str(
            np.mean(acc_inter['conv_tcja1_1_out_mean'][0]))
          + ',20,20,20,1,64,1,4,1,1\n',
          'TCJA22,'
          + str(float(layer_sparse['QuantConv_8']))
          + ',' + str(np.mean(acc_inter['conv_tcja2_1_inpt_mean'][0])) + ','
          + str(
            np.mean(acc_inter['conv_tcja2_1_out_mean'][0]))
          + ',128,128,20,1,64,1,4,1,1\n',
          'Dense1,'
          + str(float(layer_sparse['QuantDense_0']))
          + ',' + str(np.mean(acc_inter['dense1_inpt_mean'][0])) + ',' + str(
            np.mean(acc_inter['dense1_out_mean'][0]))
          + ',20,2048,512,1,1,1,1,1,1\n',
          'Dense2,' + str(float(layer_sparse['QuantDense_1']))
          + ',' + str(np.mean(acc_inter['dense2_inpt_mean'][0])) + ','
          + str(np.mean(acc_inter['dense2_out_mean'][0]))
          + ',20,512,110,1,1,1,1,1,1\n', ]

  with open('workload_' + workdir.split('/')[2]
            + '_mean.txt', 'w') as the_file:
    for li in file:
      the_file.write(li)

  file = ['name,weights,inputs,outputs,T,C,M,P,Q,R,S,HS,WS\n',
          'Conv1,'
          + str(float(layer_sparse['QuantConv_0']))
          + ',' + str(np.max(acc_inter['conv_0_inpt_min'][0])) + ',' + str(
            np.max(acc_inter['conv_0_out_min'][0]))
          + ',20,2,128,128,128,3,3,1,1\n',
          'Conv2,'
          + str(float(layer_sparse['QuantConv_1']))
          + ',' + str(np.max(acc_inter['conv_1_inpt_min'][0])) + ',' + str(
            np.max(acc_inter['conv_1_out_min'][0]))
          + ',20,128,128,64,64,3,3,1,1\n',
          'Conv3,'
          + str(float(layer_sparse['QuantConv_2']))
          + ',' + str(np.max(acc_inter['conv_2_inpt_min'][0])) + ',' + str(
            np.max(acc_inter['conv_2_out_min'][0]))
          + ',20,128,128,32,32,3,3,1,1\n',
          'Conv4,'
          + str(float(layer_sparse['QuantConv_3']))
          + ',' + str(np.max(acc_inter['conv_t_0_inpt_min'][0])) + ',' + str(
            np.max(acc_inter['conv_t_0_out_min'][0]))
          + ',20,128,128,16,16,3,3,1,1\n',
          'TCJA11,'
          + str(float(layer_sparse['QuantConv_4']))
          + ',' + str(np.max(acc_inter['conv_tcja1_0_inpt_min'][0])) + ','
          + str(
            np.max(acc_inter['conv_tcja1_0_out_min'][0]))
          + ',20,20,20,1,256,1,4,1,1\n',
          'TCJA12,'
          + str(float(layer_sparse['QuantConv_5']))
          + ',' + str(np.max(acc_inter['conv_tcja2_0_inpt_min'][0])) + ','
          + str(
            np.max(acc_inter['conv_tcja2_0_out_min'][0]))
          + ',128,128,20,1,256,1,4,1,1\n',
          'Conv5,'
          + str(float(layer_sparse['QuantConv_6']))
          + ',' + str(np.max(acc_inter['conv_t_1_inpt_min'][0])) + ',' + str(
            np.max(acc_inter['conv_t_1_out_min'][0]))
          + ',20,128,128,8,8,3,3,1,1\n',
          'TCJA21,'
          + str(float(layer_sparse['QuantConv_7']))
          + ',' + str(np.max(acc_inter['conv_tcja1_1_inpt_min'][0])) + ','
          + str(
            np.max(acc_inter['conv_tcja1_1_out_min'][0]))
          + ',20,20,20,1,64,1,4,1,1\n',
          'TCJA22,'
          + str(float(layer_sparse['QuantConv_8']))
          + ',' + str(np.max(acc_inter['conv_tcja2_1_inpt_min'][0])) + ','
          + str(
            np.max(acc_inter['conv_tcja2_1_out_min'][0]))
          + ',128,128,20,1,64,1,4,1,1\n',
          'Dense1,'
          + str(float(layer_sparse['QuantDense_0']))
          + ',' + str(np.max(acc_inter['dense1_inpt_min'][0])) + ',' + str(
            np.max(acc_inter['dense1_out_min'][0]))
          + ',20,2048,512,1,1,1,1,1,1\n',
          'Dense2,' + str(float(layer_sparse['QuantDense_1']))
          + ',' + str(np.max(acc_inter['dense2_inpt_min'][0])) + ','
          + str(np.max(acc_inter['dense2_out_min'][0]))
          + ',20,512,110,1,1,1,1,1,1\n', ]

  with open('workload_' + workdir.split('/')[2] + '_min.txt', 'w') as the_file:
    for li in file:
      the_file.write(li)

  return True


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], 'GPU')
  tf.config.experimental.set_visible_devices([], 'TPU')

  logging.info('JAX process: %d / %d',
               jax.process_index(), jax.process_count())
  logging.info('JAX local devices: %r', jax.local_devices())

  # Add a note so that we can tell which task is which JAX host.
  # (Depending on the platform task 0 is not guaranteed to be host 0)
  platform.work_unit().set_task_status(f'proc_index: {jax.process_index()}, '
                                       f'proc_count: {jax.process_count()}')
  platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY,
                                       FLAGS.workdir, 'workdir')

  train_and_evaluate(FLAGS.config, FLAGS.workdir)


if __name__ == '__main__':
  flags.mark_flags_as_required(['config', 'workdir'])
  app.run(main)
