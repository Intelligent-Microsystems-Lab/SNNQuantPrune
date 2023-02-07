# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer

"""Neuromorphic input pipeline.
"""

import jax
import jax.numpy as jnp

import tensorflow as tf
import tensorflow_datasets as tfds

from functools import partial
from flax import jax_utils


def create_input_iter(dataset_builder, config, train, cache):
  ds = create_split(dataset_builder, config, train=train, cache=cache)
  it = map(partial(prepare_tf_data, config=config), ds)
  it = jax_utils.prefetch_to_device(
      it,
      2,
      devices=jax.devices()[: config.num_devices]
      if "num_devices" in config
      else jax.devices(),
  )
  return it


def prepare_tf_data(xs, config):
  """Convert a input batch from tf Tensors to numpy arrays."""
  local_device_count = (
      config.num_devices
      if "num_devices" in config
      else jax.local_device_count()
  )

  def _prepare(x):
    # Use _numpy() for zero-copy conversion between TF and NumPy.
    x = x._numpy()  # pylint: disable=protected-access

    # reshape (host_batch_size, height, width, 3) to
    # (local_devices, device_batch_size, height, width, 3)
    return x.reshape((local_device_count, -1) + x.shape[1:])

  return jax.tree_map(_prepare, xs)


def crop_dims(tmad, low, high, dims):
  for i, d in enumerate(dims):

    idx = tf.where(tmad[:, d + 1] < high[i])
    tmad = tf.gather(tmad, idx)[:, 0, :]

    idx = tf.where(tmad[:, d + 1] >= low[i])
    tmad = tf.gather(tmad, idx)[:, 0, :]

  # Normalize
  tmad = tmad - (0, *low, 0)
  return tmad


def preprocess_data_time(addrs, times, config, wh, T, shuffle=False):
  """Preprocesses the given image for evaluation.
Args:
  image_bytes: `Tensor` representing an image binary of arbitrary size.
  dtype: data type of the image.
  image_size: image size.
Returns:
  A preprocessed image `Tensor`.
"""
  times = tf.cast(times, tf.int32)
  addrs = tf.cast(addrs, tf.int32)

  # determine start
  tbegin = tf.reduce_min(times)
  tend = tf.maximum(tbegin, tf.reduce_max(times) - 2 * T * 1000)

  start_time = tf.where(
      shuffle,
      tf.random.uniform(
          shape=(), minval=tbegin, maxval=tend + 1, dtype=tf.int32
      ),
      tf.constant(0, dtype=tf.int32),
  )

  wh = int(wh // config.resolution_scale)

  frames = tf.zeros(shape=[config.num_frames, 2, wh * wh], dtype=tf.int32)
  for i in range(config.num_frames):
    valid_idx = tf.where(
        tf.math.logical_and(
            times > (start_time + (config.time_step * i) * 1000),
            times < (start_time + (config.time_step * (i + 1)) * 1000),
        )
    )[:, 0]
    x = (
        tf.cast(tf.gather(addrs[:, 0], valid_idx), dtype=tf.float32)
        // config.resolution_scale
    )
    y = (
        tf.cast(tf.gather(addrs[:, 1], valid_idx), dtype=tf.float32)
        // config.resolution_scale
    )
    p = tf.gather(addrs[:, 2], valid_idx)
    mask = []
    mask.append(p == 0)
    mask.append(tf.math.logical_not(mask[0]))
    for j in range(2):
      position = y[mask[j]] * wh + x[mask[j]]
      events_number_per_pos = tf.math.bincount(
          tf.cast(position, tf.int32)
      )

      len_events = tf.shape(events_number_per_pos)[0]
      idx = tf.transpose(
          tf.stack(
              [
                  tf.repeat([i], len_events),
                  tf.repeat([j], len_events),
                  tf.range(0, len_events),
              ]
          )
      )
      frames = tf.tensor_scatter_nd_add(
          frames, idx, events_number_per_pos
      )

  dvs_matrix = tf.transpose(
      tf.reshape(frames, (config.num_frames, 2, wh, wh)), (0, 2, 3, 1)
  )

  model_dtype = config.dtype if "dtype" in config else jnp.float32
  dvs_matrix = tf.cast(dvs_matrix, dtype=model_dtype)

  # input scaling
  dvs_matrix /= config.time_step * config.input_scale

  return dvs_matrix


def preprocess_data_number(addrs, times, config, wh):
  """Preprocesses the given image for evaluation.
Args:
  image_bytes: `Tensor` representing an image binary of arbitrary size.
  dtype: data type of the image.
  image_size: image size.
Returns:
  A preprocessed image `Tensor`.


  Originally based on
  https://github.com/fangwei123456/spikingjelly/blob/\
  73f94ab983d0167623015537f7d4460b064cfca1/spikingjelly/datasets/utils.py
"""
  wh = int(wh // config.resolution_scale)
  frames = tf.zeros(shape=[config.num_frames, 2, wh * wh], dtype=tf.int32)

  j_l = tf.zeros(shape=[config.num_frames], dtype=tf.int32)
  j_r = tf.zeros(shape=[config.num_frames], dtype=tf.int32)
  if config.split_by == "time":
    raise NotImplementedError

  elif config.split_by == "number":
    di = tf.cast(tf.shape(times)[0] // config.num_frames, tf.int32)
    for i in range(config.num_frames):
      j_l = tf.tensor_scatter_nd_update(j_l, [[i]], [i * di])
      j_r = tf.tensor_scatter_nd_update(
          j_r,
          [[i]],
          [
              j_l[i] + di
              if i < config.num_frames - 1
              else tf.shape(times)[0]
          ],
      )
  else:
    raise NotImplementedError

  for i in range(config.num_frames):
    x = (
        tf.cast(addrs[:, 0][j_l[i]: j_r[i]], dtype=tf.float32)
        // config.resolution_scale
    )
    y = (
        tf.cast(addrs[:, 1][j_l[i]: j_r[i]], dtype=tf.float32)
        // config.resolution_scale
    )
    p = addrs[:, 2][j_l[i]: j_r[i]]
    mask = []
    mask.append(p == 0)
    mask.append(tf.math.logical_not(mask[0]))
    for j in range(2):
      position = y[mask[j]] * wh + x[mask[j]]
      events_number_per_pos = tf.math.bincount(
          tf.cast(position, tf.int32)
      )

      len_events = tf.shape(events_number_per_pos)[0]
      idx = tf.transpose(
          tf.stack(
              [
                  tf.repeat([i], len_events),
                  tf.repeat([j], len_events),
                  tf.range(0, len_events),
              ]
          )
      )
      frames = tf.tensor_scatter_nd_add(
          frames, idx, events_number_per_pos
      )

  dvs_matrix = tf.transpose(
      tf.reshape(frames, (config.num_frames, 2, wh, wh)), (0, 2, 3, 1)
  )

  model_dtype = config.dtype if "dtype" in config else jnp.float32
  dvs_matrix = tf.cast(dvs_matrix, dtype=model_dtype)
  return dvs_matrix


def create_split(dataset_builder, config, train, cache):
  """Creates a split from a Neuromorphic dataset using TensorFlow Datasets.
Args:
  dataset_builder: TFDS dataset builder for Neuromorphic Data.
  batch_size: the batch size returned by the data pipeline.
  train: Whether to load the train or evaluation split.
  dtype: data type of the image.
  image_size: The target size of the images.
  cache: Whether to cache the dataset.
Returns:
  A `tf.data.Dataset`.
"""
  if config.dataset == "dvs_gesture_tfds":
    base_size = 128
  elif config.dataset == "nmnist_tfds":
    base_size = 34
  elif config.dataset == "asl_dvs_tfds":
    base_size = 240
  else:
    raise Exception("Unknown dataset: " + config.dataset)

  if train:
    train_examples = dataset_builder.info.splits["train"].num_examples
    split_size = train_examples // jax.process_count()
    start = jax.process_index() * split_size
    split = "train[{}:{}]".format(start, start + split_size)
  else:
    validate_examples = dataset_builder.info.splits["test"].num_examples
    split_size = validate_examples // jax.process_count()
    start = jax.process_index() * split_size
    split = "test[{}:{}]".format(start, start + split_size)

  def decode_example(example):
    if config.split_by == "time":
      if train:
        dvs_matrix = preprocess_data_time(
            example["addrs"],
            example["times"],
            config,
            base_size,
            T=config.train_chunk_size,
            shuffle=True,
        )
      else:
        dvs_matrix = preprocess_data_time(
            example["addrs"],
            example["times"],
            config,
            base_size,
            T=config.test_chunk_size,
            shuffle=True,
        )
    if config.split_by == "number":
      dvs_matrix = preprocess_data_number(
          example["addrs"], example["times"], config, base_size
      )

    return {
        "dvs_matrix": dvs_matrix,
        "label": tf.cast(example["label"], tf.int8),
    }

  ds = dataset_builder.as_dataset(
      split=split,
      decoders={
          "addrs": tfds.decode.SkipDecoding(),
          "times": tfds.decode.SkipDecoding(),
      },
  )

  # Debug
  # test  = next(iter(ds))
  # dvs_matrix = preprocess_data_time( test['addrs'], test['times'], config,
  # base_size,  T=config.train_chunk_size, shuffle=True)

  options = tf.data.Options()
  options.experimental_threading.private_threadpool_size = 48
  ds = ds.with_options(options)

  if cache:
    ds = ds.cache()

  if train:
    ds = ds.repeat()
    ds = ds.shuffle(16 * config.batch_size // jax.process_count(), seed=0)

  ds = ds.map(
      decode_example, num_parallel_calls=tf.data.experimental.AUTOTUNE
  )
  if train:
    ds = ds.batch(
        config.batch_size // jax.process_count(), drop_remainder=True
    )
  else:
    ds = ds.batch(
        config.eval_batch_size // jax.process_count(), drop_remainder=True
    )

  if not train:
    ds = ds.repeat()

  ds = ds.prefetch(10)

  return ds
