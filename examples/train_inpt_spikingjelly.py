# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Originally copied from https://github.com/google/flax/tree/main/examples

"""Training script.
The data is loaded using tensorflow_datasets.
"""


import functools
import subprocess
import time

from absl import app
from absl import flags
from absl import logging

from clu import platform
from clu import metric_writers
from clu import periodic_actions

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
    restore_checkpoint,
    save_checkpoint,
    create_learning_rate_fn,
    create_model,
    create_train_state,
    train_step,
    eval_step,
    sync_batch_stats,
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

  writer_train = metric_writers.create_default_writer(
      logdir=workdir + '/train', just_logging=jax.process_index() != 0)
  writer_eval = metric_writers.create_default_writer(
      logdir=workdir + '/eval', just_logging=jax.process_index() != 0)

  logging.get_absl_handler().use_absl_log_file('absl_logging', FLAGS.workdir)
  logging.info('Git commit: ' + subprocess.check_output(
      ['git', 'rev-parse', 'HEAD']).decode('ascii').strip())
  logging.info(config)

  rng = random.PRNGKey(config.seed)

  if config.batch_size % jax.device_count() > 0:
    raise ValueError('Batch size (' + str(config.batch_size) + ') must be \
      divisible by the number of devices (' + str(jax.device_count()) + ').')

  train_set = DVS128Gesture('/home/clemens/datasets/DVS128Gesture', train=True,
                            data_type='frame', split_by='number',
                            frames_number=config.num_frames)
  test_set = DVS128Gesture('/home/clemens/datasets/DVS128Gesture', train=False,
                           data_type='frame', split_by='number',
                           frames_number=config.num_frames)

  train_iter = DataLoader(
      dataset=train_set,
      batch_size=config.batch_size,
      shuffle=True,
      num_workers=16,
      drop_last=True,
      pin_memory=True)

  eval_iter = DataLoader(
      dataset=test_set,
      batch_size=config.eval_batch_size,
      shuffle=False,
      num_workers=16,
      drop_last=False,
      pin_memory=True)

  dev_num = config.num_devices if 'num_devices' in config else\
      jax.local_device_count()

  steps_per_checkpoint = 10

  base_learning_rate = config.learning_rate
  model_dtype = config.dtype if 'dtype' in config else jnp.float32

  model_cls = getattr(models, config.model)

  model = create_model(
      model_cls=model_cls, num_classes=config.num_classes,
      model_dtype=model_dtype, config=config)

  learning_rate_fn = create_learning_rate_fn(
      config, base_learning_rate, len(train_set.samples) // config.batch_size)

  image_size = next(iter(train_iter))[0].shape[-2]
  rng, subkey = jax.random.split(rng, 2)
  state = create_train_state(
      subkey, config, model, image_size, learning_rate_fn)

  if config.pretrained:
    state = model.load_model_fn(state, config.pretrained)

    def update_prune_mask(x):
      if type(x) is not dict:
        return x
      else:
        mask = np.ones(x['kernel'].shape)
        k = int(np.prod(x['kernel'].shape) * config.prune_percentage)
        idx = np.argpartition(np.abs(x['kernel']).reshape(-1), k)[:k]
        mask.reshape(-1)[idx] = 0

        return {'prune_0': {'mask': jnp.array(mask)}, 'kernel': x['kernel'],
                'DuQ_0': x['DuQ_0']}

    def update_quant_params(x):
      if type(x) is not dict:
        return x
      else:
        return {'DuQ_0': {'a':
                          jnp.array((config.quant.init_fn(
                              x['kernel'], bits=config.quant.bits, sign=True),
                          )),
                          'c':
                          jnp.array((config.quant.init_fn(
                              x['kernel'], bits=config.quant.bits, sign=True),
                          ))},
                'prune_0': x['prune_0'],
                'kernel': x['kernel']}

    global_weights = []
    global global_mask

    def create_global_weights(x):
      if type(x) is not dict:
        return x
      else:
        global_weights.append(x['kernel'])
        return x

    def get_global_mask(x):
      if type(x) is not dict:
        return x
      else:
        global global_mask
        local_mask = global_mask[:np.prod(x['kernel'].shape)]
        global_mask = global_mask[np.prod(x['kernel'].shape):]
        return {'prune_0': {'mask': jnp.reshape(local_mask,
                                                x['kernel'].shape)},
                'DuQ_0': x['DuQ_0'],
                'kernel': x['kernel']}

    def check_quant_obj(x):
      if type(x) is not dict:
        return True
      if 'kernel' in x.keys():
        return True
      return False

    if config.quant.prune_percentage > 0.:
      # pruning local - layerwise
      if config.quant.prune_global is False:
        state.params['params'] = jax.tree_map(
            update_prune_mask, state.params['params'], is_leaf=check_quant_obj)
      # pruning global
      if config.quant.prune_global is True:
        _ = jax.tree_map(
            create_global_weights, state.params['params'],
            is_leaf=check_quant_obj)

        tmp_gw = jnp.concatenate([jnp.reshape(x, (-1))
                                  for x in global_weights])

        global_mask = np.ones(tmp_gw.shape)
        k = int(np.prod(tmp_gw.shape) * config.quant.prune_percentage)
        idx = np.argpartition(np.abs(tmp_gw), k)[:k]
        global_mask[idx] = 0

        state.params['params'] = jax.tree_map(
            get_global_mask, state.params['params'], is_leaf=check_quant_obj)

    if config.quant.start_epoch == -1:
      logging.info('Quantization start.')
      state.params['params'] = jax.tree_map(
          update_quant_params, state.params['params'], is_leaf=check_quant_obj)

    logging.info('Successfully restored model from: ' + config.pretrained)

  state = restore_checkpoint(state, workdir)
  # step_offset > 0 if restarting from checkpoint
  step_offset = int(state.step)

  # Log number of parameters
  total_num_params = jnp.sum(jnp.array(jax.tree_util.tree_flatten(
      jax.tree_util.tree_map(lambda x: jnp.prod(jnp.array(x.shape)),
                             state.params))[0]))
  logging.info('Total number of parameters: ' + str(total_num_params))

  state = jax_utils.replicate(state, devices=jax.devices(
  )[:config.num_devices] if 'num_devices' in config else jax.devices())
  # Debug note:
  # 1. Make above line a comment "state = jax_utils.replicate(state)".
  # 2. In train_util.py make all pmean commands comments.
  # 3. Use debug train_step.
  # 4. Uncomment JIT configs at the top.
  # 5. Commted before training eval.

  p_train_step = jax.pmap(
      functools.partial(
          train_step,
          learning_rate_fn=learning_rate_fn,
          weight_decay=config.weight_decay,
          smoothing=config.smoothing,
          loss_type=config.loss_fn,
          online=config.online if 'online' in config else False,
          burnin=config.online if 'burnin' in config else False,
      ),
      axis_name='batch',
      devices=jax.devices()[
          :config.num_devices] if 'num_devices' in config else jax.devices(),
  )

  p_eval_step = jax.pmap(
      functools.partial(
          eval_step,
          loss_type=config.loss_fn,
          smoothing=config.smoothing,
          burnin=config.online if 'burnin' in config else False,
      ),
      axis_name='batch',
      devices=jax.devices()[
          :config.num_devices] if 'num_devices' in config else jax.devices(),
  )

  # # Debug
  # p_train_step = functools.partial(
  #         train_step,
  #         learning_rate_fn=learning_rate_fn,
  #         weight_decay=config.weight_decay,
  #         smoothing=config.smoothing,
  #         loss_type=config.loss_fn,
  #         online=config.online if 'online' in config else False,
  #         burnin=config.online if 'burnin' in config else False,
  #     )
  # p_eval_step = functools.partial(
  #         eval_step,
  #         loss_type=config.loss_fn,
  #         smoothing=config.smoothing,
  #         burnin=config.online if 'burnin' in config else False,
  #     )

  # evaluate model before training
  eval_metrics = []
  for frame, label in eval_iter:
    rng_list = jax.random.split(rng, dev_num + 1)
    rng = rng_list[0]
    eval_batch = {'dvs_matrix': jnp.reshape(jnp.transpose(frame.numpy(),
                                                          (0, 1, 3, 4, 2)),
                                            (config.num_devices, -1,
                                             config.num_frames, image_size,
                                             image_size, 2)), 'label':
                  jnp.reshape(label.numpy(), (config.num_devices, -1))}

    metrics = p_eval_step(state, eval_batch, rng_list[1:])
    eval_metrics.append(metrics)

  logging.info('density:')
  logging.info(jax.tree_map(lambda x: jnp.sum(x != 0)
                            / np.prod(x.shape), state.params['params']))

  eval_metrics = common_utils.stack_forest(eval_metrics)
  summary = jax.tree_map(lambda x: x.mean(), eval_metrics)
  logging.info('Before training eval - loss: %.4f, accuracy: %.2f',
               summary['loss'], summary['accuracy'] * 100)

  train_metrics = []
  hooks = []
  eval_best = 0.
  latency = -1.
  if jax.process_index() == 0:
    hooks += [periodic_actions.Profile(num_profile_steps=5, logdir=workdir)]
  train_metrics_last_t = time.time()
  logging.info('Initial compilation, this might take some minutes...')
  for step in range(config.num_epochs):

    if step == config.quant.start_epoch:
      if 'DuQ' in str(config.quant.weight):
        logging.info('Quantization start.')
        state = jax_utils.unreplicate(state)
        # quantization
        state.params['params'] = jax.tree_map(
            update_quant_params, state.params['params'],
            is_leaf=check_quant_obj)
        state = jax_utils.replicate(state, devices=jax.devices(
        )[:config.num_devices] if 'num_devices' in config else jax.devices())

    for frame, label in train_iter:

      rng_list = jax.random.split(rng, dev_num + 1)
      rng = rng_list[0]

      batch = {'dvs_matrix': jnp.reshape(jnp.transpose(frame.numpy(),
                                                       (0, 1, 3, 4, 2)),
                                         (config.num_devices, -1,
                                          config.num_frames, image_size,
                                          image_size, 2)), 'label':
               jnp.reshape(label.numpy(), (config.num_devices, -1))}

      state, metrics = p_train_step(state, batch, rng_list[1:])
      # # Debug
      # state, metrics = p_train_step(
      #     state, {'dvs_matrix': batch['dvs_matrix'][0, :, :, :, :, :],
      #             'label': batch['label'][0]}, rng_list[2])

    for h in hooks:
      h(step)
    if step == step_offset:
      logging.info('Initial compilation completed.')

    if config.get('log_every_steps'):
      train_metrics.append(metrics)
      if (step + 1) % config.log_every_steps == 0:
        train_metrics = common_utils.stack_forest(train_metrics)
        summary = {
            f'{k}': v
            for k, v in jax.tree_map(lambda x: x.mean(), train_metrics).items()
        }
        summary['steps_per_second'] = config.log_every_steps / (
            time.time() - train_metrics_last_t)
        summary['epoch'] = step
        writer_train.write_scalars(step + 1, summary)
        writer_train.flush()
        train_metrics = []
        train_metrics_last_t = time.time()

    eval_metrics = []
    # sync batch statistics across replicas
    state = sync_batch_stats(state)
    eval_latency = []
    for frame, label in eval_iter:
      rng_list = jax.random.split(rng, dev_num + 1)
      rng = rng_list[0]
      eval_batch = {'dvs_matrix': jnp.reshape(jnp.transpose(frame.numpy(),
                                                            (0, 1, 3, 4, 2)), (
          config.num_devices, -1, config.num_frames, image_size, image_size,
          2)),
          'label': jnp.reshape(label.numpy(), (config.num_devices, -1))}
      start_t = time.time()
      metrics = p_eval_step(state, eval_batch, rng_list[1:])
      eval_latency.append(time.time() - start_t)
      eval_metrics.append(metrics)

    # discard first latency - commpile time...
    latency = np.mean(eval_latency[1:])
    eval_metrics = common_utils.stack_forest(eval_metrics)
    summary = jax.tree_map(lambda x: x.mean(), eval_metrics)
    logging.info('eval epoch: %d, loss: %.4f, accuracy: %.2f',
                 step, summary['loss'], summary['accuracy'] * 100)
    writer_eval.write_scalars(
        step + 1, {f'{key}': val for key, val in summary.items()})
    writer_eval.flush()
    if summary['accuracy'] > eval_best and step > config.quant.start_epoch:
      save_checkpoint(state, workdir + '/best')
      logging.info('!!!! Saved new best model with accuracy %.4f',
                   summary['accuracy'])
      eval_best = summary['accuracy']

    if (step + 1) % steps_per_checkpoint == 0:
      state = sync_batch_stats(state)
      save_checkpoint(state, workdir)

  logging.info('density:')
  logging.info(jax.tree_map(lambda x: jnp.sum(x != 0)
                            / np.prod(x.shape), state.params['params']))

  # Wait until computations are done before exiting
  jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

  logging.info('Total number of parameters: ' + str(total_num_params))
  logging.info('Best accuracy: %.4f with latency %.4f', eval_best, latency)
  return state


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
