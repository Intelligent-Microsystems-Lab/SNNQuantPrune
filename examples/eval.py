# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer

"""Eval script.
The data is loaded using tensorflow_datasets.
"""


import functools

from absl import app
from absl import flags
from absl import logging

from clu import platform

from flax import jax_utils
from flax.training import common_utils

import jax
from jax import random

import tensorflow as tf
import tensorflow_datasets as tfds

import input_pipeline
import models

import ml_collections
from ml_collections import config_flags


from train_utils import (
    TrainState,
    restore_checkpoint,
    create_model,
    create_train_state,
    eval_step,
)


FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", None, "Directory to store model data.")
config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)


def evaluate(config: ml_collections.ConfigDict, workdir: str) -> TrainState:
  """Execute model training and evaluation loop.
Args:
  config: Hyperparameter configuration for training and evaluation.
  workdir: Directory where the tensorboard summaries are written to.
Returns:
  Final TrainState.
"""

  rng = random.PRNGKey(config.seed)

  if config.batch_size % jax.device_count() > 0:
    raise ValueError(
        "Batch size ("
        + str(config.batch_size)
        + ") must be \
      divisible by the number of devices ("
        + str(jax.device_count())
        + ")."
    )

  # Load Data
  dataset_builder = tfds.builder(config.dataset)
  dataset_builder.download_and_prepare()

  eval_iter = input_pipeline.create_input_iter(
      dataset_builder, config, train=False, cache=config.cache
  )
  train_iter = input_pipeline.create_input_iter(
      dataset_builder, config, train=True, cache=config.cache
  )

  if config.steps_per_eval == -1:
    num_validation_examples = dataset_builder.info.splits[
        "test"
    ].num_examples
    steps_per_eval = num_validation_examples // config.eval_batch_size
  else:
    steps_per_eval = config.steps_per_eval

  model_cls = getattr(models, config.model)
  model = create_model(
      model_cls=model_cls, num_classes=config.num_classes, config=config
  )

  image_size = next(train_iter)["dvs_matrix"].shape[-2]
  rng, subkey = jax.random.split(rng, 2)
  state = create_train_state(
      subkey, config, model, image_size, lambda x: 0.0
  )

  state = restore_checkpoint(state, workdir)

  state = jax_utils.replicate(state)

  p_eval_step = jax.pmap(
      functools.partial(
          eval_step,
          loss_type=config.loss_fn,
          smoothing=config.smoothing,
          burnin=config.online if "burnin" in config else False,
      ),
      axis_name="batch",
  )

  # evaluate model before training
  eval_metrics = []
  for _ in range(steps_per_eval):
    rng_list = jax.random.split(rng, jax.local_device_count() + 1)
    rng = rng_list[0]

    eval_batch = next(eval_iter)
    metrics = p_eval_step(state, eval_batch, rng_list[1:])
    eval_metrics.append(metrics)

  eval_metrics = common_utils.stack_forest(eval_metrics)
  summary = jax.tree_map(lambda x: x.mean(), eval_metrics)
  logging.info(
      "Eval loss: %.4f, accuracy: %.2f",
      summary["loss"],
      summary["accuracy"] * 100,
  )

  # Wait until computations are done before exiting
  jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

  return state


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], "GPU")
  tf.config.experimental.set_visible_devices([], "TPU")

  logging.info(
      "JAX process: %d / %d", jax.process_index(), jax.process_count()
  )
  logging.info("JAX local devices: %r", jax.local_devices())

  # Add a note so that we can tell which task is which JAX host.
  # (Depending on the platform task 0 is not guaranteed to be host 0)
  platform.work_unit().set_task_status(
      f"proc_index: {jax.process_index()}, "
      f"proc_count: {jax.process_count()}"
  )
  platform.work_unit().create_artifact(
      platform.ArtifactType.DIRECTORY, FLAGS.workdir, "workdir"
  )

  evaluate(FLAGS.config, FLAGS.workdir)


if __name__ == "__main__":
  flags.mark_flags_as_required(["config", "workdir"])
  app.run(main)
