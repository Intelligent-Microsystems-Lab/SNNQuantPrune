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
from jax import random

import tensorflow as tf
import tensorflow_datasets as tfds

import numpy as np
import jax.numpy as jnp

import input_pipeline
import models

import ml_collections
from ml_collections import config_flags


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

  # config.unlock()
  # config.num_frames = int(config.train_chunk_size / config.time_step)
  # config.lock()

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

  # Load Data
  dataset_builder = tfds.builder(config.dataset, data_dir=config.tfds_data_dir)
  dataset_builder.download_and_prepare()

  train_iter = input_pipeline.create_input_iter(
      dataset_builder, config, train=True, cache=config.cache)
  eval_iter = input_pipeline.create_input_iter(
      dataset_builder, config, train=False, cache=config.cache)

  dev_num = config.num_devices if 'num_devices' in config else\
      jax.local_device_count()

  steps_per_epoch = (
      dataset_builder.info.splits['train'].num_examples // config.batch_size
  )

  if config.num_train_steps == -1:
    num_steps = int(steps_per_epoch * config.num_epochs)
  else:
    num_steps = config.num_train_steps

  if config.steps_per_eval == -1:
    num_validation_examples = dataset_builder.info.splits['test'].num_examples
    steps_per_eval = num_validation_examples // config.eval_batch_size
  else:
    steps_per_eval = config.steps_per_eval

  steps_per_checkpoint = steps_per_epoch * 10

  base_learning_rate = config.learning_rate
  model_dtype = config.dtype if 'dtype' in config else jnp.float32

  model_cls = getattr(models, config.model)
  # if 'partial' not in model_cls.__doc__:
  #   # raw res conv net for network sweeps
  #   model_cls = functools.partial(model_cls, depth=config.depth,
  #                                 width=config.width,
  #                                 num_compartments=config.num_compartments)

  model = create_model(
      model_cls=model_cls, num_classes=config.num_classes,
      model_dtype=model_dtype, config=config)

  learning_rate_fn = create_learning_rate_fn(
      config, base_learning_rate, steps_per_epoch
      * (config.num_frames if 'online' in config else 1.))

  image_size = next(train_iter)['dvs_matrix'].shape[-2]
  rng, subkey = jax.random.split(rng, 2)
  state = create_train_state(
      subkey, config, model, image_size, learning_rate_fn)

  if config.pretrained:
    state = model.load_model_fn(state, config.pretrained)
    logging.info('Successfully restored model from: ' + config.pretrained)

  state = restore_checkpoint(state, workdir)
  # step_offset > 0 if restarting from checkpoint
  step_offset = int(state.step)

  # Log number of parameters
  total_num_params = jnp.sum(jnp.array(jax.tree_util.tree_flatten(
      jax.tree_util.tree_map(lambda x: jnp.prod(jnp.array(x.shape)),
                             state.params))[0]))
  logging.info('Total number of parameters: ' + str(total_num_params))

  # Log number of FLOPS
  @jax.jit
  def fwd_pass_model(weights, batch_stats, inputs):
    variables = {'params': weights,
                 'batch_stats': batch_stats, }
    (logits, _), _ = state.apply_fn(
        variables,
        inputs,
        trgt=None,
        train=False,
        online=False,
        rng=random.PRNGKey(0),
        mutable=['batch_stats'],
        rngs={'dropout': random.PRNGKey(0)},
    )
    return logits

  # m = jax.xla_computation(fwd_pass_model)(state.params['params'],
  #                                         state.batch_stats, jnp.ones(
  #     (1, config.num_frames, image_size, image_size, 2))).as_hlo_module()
  # client = jax.lib.xla_bridge.get_backend()
  # cost = jax.lib.xla_client._xla.hlo_module_cost_analysis(client, m)
  # logging.info('Total number of FLOPs (fwd): ' + str(cost['flops']))

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
  for _ in range(steps_per_eval):
    rng_list = jax.random.split(rng, dev_num + 1)
    rng = rng_list[0]

    eval_batch = next(eval_iter)
    metrics = p_eval_step(state, eval_batch, rng_list[1:])
    eval_metrics.append(metrics)

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
  for step, batch in zip(range(step_offset, num_steps), train_iter):
    rng_list = jax.random.split(rng, dev_num + 1)
    rng = rng_list[0]
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
        writer_train.write_scalars(step + 1, summary)
        writer_train.flush()
        train_metrics = []
        train_metrics_last_t = time.time()

    if (step + 1) % steps_per_epoch == 0:
      epoch = (step + 1) // steps_per_epoch
      eval_metrics = []

      # sync batch statistics across replicas
      state = sync_batch_stats(state)
      eval_latency = []
      for _ in range(steps_per_eval):
        rng_list = jax.random.split(rng, dev_num + 1)
        rng = rng_list[0]

        eval_batch = next(eval_iter)
        start_t = time.time()
        metrics = p_eval_step(state, eval_batch, rng_list[1:])
        eval_latency.append(time.time() - start_t)
        eval_metrics.append(metrics)

      # discard first latency - commpile time...
      latency = np.mean(eval_latency[1:])
      eval_metrics = common_utils.stack_forest(eval_metrics)
      summary = jax.tree_map(lambda x: x.mean(), eval_metrics)
      logging.info('eval epoch: %d, loss: %.4f, accuracy: %.2f',
                   epoch, summary['loss'], summary['accuracy'] * 100)
      writer_eval.write_scalars(
          step + 1, {f'{key}': val for key, val in summary.items()})
      writer_eval.flush()
      if summary['accuracy'] > eval_best:
        save_checkpoint(state, workdir + '/best')
        logging.info('!!!! Saved new best model with accuracy %.4f',
                     summary['accuracy'])
        eval_best = summary['accuracy']

    if (step + 1) % steps_per_checkpoint == 0 or step + 1 == num_steps:
      state = sync_batch_stats(state)
      save_checkpoint(state, workdir)

  # Wait until computations are done before exiting
  jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

  # logging.info('Total number of FLOPs (fwd): ' + str(cost['flops']))
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
