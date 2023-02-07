# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Originally copied from https://github.com/google/flax/tree/main/examples

"""Training utils
"""

import tree
from typing import Any
import math

from flax.training import train_state
from flax.training import checkpoints
from flax.training import common_utils

import jax
import jax.numpy as jnp
from jax import lax

import optax


import ml_collections


class TrainState(train_state.TrainState):
  batch_stats: Any


def restore_checkpoint(state, workdir):
  return checkpoints.restore_checkpoint(workdir, state)


def save_checkpoint(state, workdir):
  if jax.process_index() == 0:
    # get train state from the first replica
    state = jax.device_get(jax.tree_map(lambda x: x[0], state))
    step = int(state.step)
    checkpoints.save_checkpoint(
        workdir, state, step, keep=3, overwrite=True
    )


def create_learning_rate_fn(
    config: ml_collections.ConfigDict,
    base_learning_rate: float,
    steps_per_epoch: int,
):
  """Create learning rate schedule."""
  if "lr_boundaries_scale" in config:
    schedule_fn = optax.piecewise_constant_schedule(
        config.learning_rate,
        {
            int(k) * steps_per_epoch: v
            for k, v in config.lr_boundaries_scale.items()
        },
    )
  elif "t_max" in config:
    schedule_fn = optax.sgdr_schedule(
        [
            {
                "decay_steps": config.t_max * steps_per_epoch,
                "init_value": base_learning_rate,
                "peak_value": base_learning_rate,
                "warmup_steps": 0,
            }
        ]
        * math.ceil(config.num_epochs / config.t_max)
    )
  elif config.quant.start_epoch > 0:
    num_e1 = config.quant.start_epoch
    cosine_epochs1 = max(num_e1 - config.warmup_epochs, 1)
    cosine_fn1 = optax.cosine_decay_schedule(
        init_value=base_learning_rate,
        decay_steps=cosine_epochs1 * steps_per_epoch,
    )

    num_e2 = config.num_epochs - config.quant.start_epoch
    cosine_epochs2 = max(num_e2 - config.warmup_epochs, 1)
    cosine_fn2 = optax.cosine_decay_schedule(
        init_value=base_learning_rate,
        decay_steps=cosine_epochs2 * steps_per_epoch,
    )

    if config.warmup_epochs != 0.0:
      warmup_fn1 = optax.linear_schedule(
          init_value=0.0,
          end_value=base_learning_rate,
          transition_steps=config.warmup_epochs * steps_per_epoch,
      )
      warmup_fn2 = optax.linear_schedule(
          init_value=0.0,
          end_value=base_learning_rate,
          transition_steps=config.warmup_epochs * steps_per_epoch,
      )
      b_len = [config.warmup_epochs * steps_per_epoch,
               (config.quant.start_epoch - config.warmup_epochs)
               * steps_per_epoch, config.warmup_epochs * steps_per_epoch]
      b_len[1] += b_len[0]
      b_len[2] += b_len[1]

      schedule_fn = optax.join_schedules(
          schedules=[warmup_fn1, cosine_fn1, warmup_fn2, cosine_fn2],
          boundaries=b_len
      )
    else:
      schedule_fn = schedule_fn = optax.join_schedules(
          schedules=[cosine_fn1, cosine_fn2],
          boundaries=[config.quant.start_epoch * steps_per_epoch]
      )
  else:
    cosine_epochs = max(config.num_epochs - config.warmup_epochs, 1)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=base_learning_rate,
        decay_steps=cosine_epochs * steps_per_epoch,
    )

    if config.warmup_epochs != 0.0:
      warmup_fn = optax.linear_schedule(
          init_value=0.0,
          end_value=base_learning_rate,
          transition_steps=config.warmup_epochs * steps_per_epoch,
      )
      schedule_fn = optax.join_schedules(
          schedules=[warmup_fn, cosine_fn],
          boundaries=[config.warmup_epochs * steps_per_epoch]
      )
    else:
      schedule_fn = cosine_fn
  return schedule_fn


def create_model(*, model_cls, num_classes, model_dtype, **kwargs):
  return model_cls(num_classes=num_classes, dtype=model_dtype, **kwargs)


def initialized(key, image_size, model, t):
  if t == -1:
    input_shape = (1, image_size, image_size, 2)
  else:
    input_shape = (1, t, image_size, image_size, 2)
  key, rng, prng = jax.random.split(key, 3)

  @jax.jit
  def init(*args):
    return model.init(
        *args,
        rng=rng,
        trgt=jnp.ones((1,)),
        online=True if t == -1 else False,
        train=False
    )

  variables = init(
      {"params": key, "dropout": prng}, jnp.zeros(input_shape, model.dtype)
  )

  return variables["params"], variables["batch_stats"]


def create_train_state(
    rng, config: ml_collections.ConfigDict, model, image_size, learning_rate_fn
):
  """Create initial training state."""

  params, batch_stats = initialized(
      rng, image_size, model, -1 if "online" in config else config.num_frames
  )
  if config.optimizer == "rmsprop":
    tx = optax.rmsprop(
        learning_rate=learning_rate_fn,
        decay=0.9,
        momentum=config.momentum,
        eps=0.001,
    )
  elif config.optimizer == "sgd":
    tx = optax.sgd(
        learning_rate=learning_rate_fn,
        momentum=config.momentum,
        nesterov=config.nesterov,
    )
  elif config.optimizer == "adam":
    tx = optax.adam(learning_rate=learning_rate_fn)
  else:
    raise Exception("Unknown optimizer in config: " + config.optimizer)

  state = TrainState.create(
      apply_fn=model.apply,
      params={"params": params},
      tx=tx,
      batch_stats=batch_stats,
  )
  return state


def cross_entropy_loss(logits, labels, smoothing=0):
  one_hot_labels = common_utils.onehot(labels, num_classes=logits.shape[1])

  factor = smoothing
  one_hot_labels *= 1 - factor
  one_hot_labels += factor / one_hot_labels.shape[1]

  xentropy = jnp.mean(
      optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels)
  )

  return xentropy


def decolle_loss(logits, labels, smoothing=0, T=1):
  one_hot_labels = common_utils.onehot(labels, num_classes=logits.shape[1])

  factor = smoothing
  one_hot_labels *= 1 - factor
  one_hot_labels += factor / one_hot_labels.shape[1]

  # smooth L1 loss
  smoothl1 = jnp.mean(
      optax.huber_loss(predictions=logits / T, targets=one_hot_labels)
  )

  return smoothl1


def mse_loss(logits, labels, smoothing=0, T=1):
  one_hot_labels = common_utils.onehot(labels, num_classes=logits.shape[1])

  factor = smoothing
  one_hot_labels *= 1 - factor
  one_hot_labels += factor / one_hot_labels.shape[1]

  return jnp.mean(jnp.square(logits / T - one_hot_labels))


def compute_metrics(logits, labels, smoothing, loss_fn):
  loss = loss_fn(logits, labels, smoothing)
  accuracy = jnp.argmax(logits, -1) == labels
  metrics = {"loss": loss, "accuracy": accuracy}

  return metrics


def weight_decay_fn(params):
  l2_params = [
      p
      for ((mod_name), p) in tree.flatten_with_path(params)
      if "BatchNorm" not in str(mod_name) and "bn_init" not in str(mod_name)
  ]
  return 0.5 * sum(jnp.sum(jnp.square(p)) for p in l2_params)


# pmean only works inside pmap because it needs an axis name.
# This function will average the inputs across all devices.
cross_replica_mean = jax.pmap(lambda x: lax.pmean(x, "x"), "x")


def sync_batch_stats(state):
  """Sync the batch statistics across replicas."""
  # Each device has its own version of the running average batch statistics and
  # we sync them before evaluation.
  return state.replace(batch_stats=cross_replica_mean(state.batch_stats))


def train_step(
    state,
    batch,
    rng,
    learning_rate_fn,
    weight_decay,
    smoothing,
    loss_type,
    online=False,
    burnin=0,
    return_grads=False,
):
  """Perform a single training step."""
  rng, prng = jax.random.split(rng, 2)

  def loss_fn(params, inputs, targets, u_state=None):
    """loss function used for training."""
    (logits, u_state), new_model_state = state.apply_fn(
        {"params": params, "batch_stats": state.batch_stats},
        inputs,
        trgt=targets,
        train=True,
        u_state=u_state,
        online=online,
        mutable=["batch_stats"],
        rng=prng,
        rngs={"dropout": rng},
    )

    loss = loss_type(logits, targets, smoothing)
    loss += weight_decay * weight_decay_fn(params)

    return loss, (logits, u_state, new_model_state)

  if online:
    # TODO: @clee1994 think about random keys...
    # online learning

    def one_step_fn(carry, x):
      u, state, step = carry

      grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
      aux, grads = grad_fn(state.params["params"], x, batch["label"], u)
      grads = lax.pmean(grads, axis_name="batch")

      grads = jax.tree_util.tree_map(
          lambda x: x * (step >= burnin), grads
      )
      new_state = state.apply_gradients(
          grads={"params": grads}, batch_stats=aux[1][2]["batch_stats"]
      )

      return (aux[1][1], new_state, step + 1), (aux[1][0], grads)

    _, u_state = state.apply_fn(
        {
            "params": state.params["params"],
            "batch_stats": state.batch_stats,
        },
        batch["dvs_matrix"][:, 0, :],
        trgt=batch["label"],
        train=False,
        rng=prng,
        online=True,
        rngs={"dropout": rng},
    )
    # init state to zero
    u_state = [[jnp.zeros_like(x) for x in y] for y in u_state]
    u_state = [tuple(x) for x in u_state]

    lr_start = learning_rate_fn(state.step)

    inpt_prep = jnp.moveaxis(
        batch["dvs_matrix"], (0, 1, 2, 3, 4), (1, 0, 2, 3, 4)
    )
    (_, new_state, _), (logits, grads) = jax.lax.scan(
        one_step_fn, (u_state, state, 0), inpt_prep
    )

    # only return last gradient - intended for unit tests
    grads = jax.tree_util.tree_map(lambda x: x[-1, :], grads)

    # compute metrics
    metrics = compute_metrics(
        jnp.mean(logits[burnin:, :], axis=0),
        batch["label"],
        smoothing,
        loss_type,
    )
    metrics["learning_rate"] = (
        learning_rate_fn(state.step) + lr_start
    ) / 2

  else:
    # offline learning
    lr = learning_rate_fn(state.step)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(
        state.params["params"], batch["dvs_matrix"], batch["label"]
    )

    # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
    grads = lax.pmean(grads, axis_name="batch")

    # TODO @clee1994 consider burnin
    metrics = compute_metrics(
        aux[1][0], batch["label"], smoothing, loss_type
    )
    metrics["learning_rate"] = lr
    metrics["logits"] = aux[1][0]
    new_state = state.apply_gradients(
        grads={"params": grads}, batch_stats=aux[1][2]["batch_stats"]
    )

  if return_grads:
    return new_state, metrics, grads

  return new_state, metrics


def eval_step(state, batch, rng, smoothing, loss_type, burnin=0):
  rng, prng = jax.random.split(rng, 2)

  variables = {
      "params": state.params["params"],
      "batch_stats": state.batch_stats,
  }
  (logits, _), _ = state.apply_fn(
      variables,
      batch["dvs_matrix"],
      trgt=batch["label"],
      train=False,
      online=False,
      rng=prng,
      mutable=["batch_stats"],
      rngs={"dropout": rng},
  )
  metrics = compute_metrics(logits, batch["label"], smoothing, loss_type)
  metrics["accuracy"] = metrics["accuracy"]

  return metrics
