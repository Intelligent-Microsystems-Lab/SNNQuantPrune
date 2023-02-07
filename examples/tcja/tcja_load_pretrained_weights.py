# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer

import jax
from flax.core import unfreeze, freeze
from flax.training import train_state
from flax.training import checkpoints

from typing import Any
import jax.numpy as jnp

import torch


class TrainState(train_state.TrainState):
  batch_stats: Any


torch_map = {
    "conv.0.0": "QuantConv_0",
    "conv.0.1": "BatchNorm_0",
    "conv.3.0": "QuantConv_1",
    "conv.3.1": "BatchNorm_1",
    "conv.6.0": "QuantConv_2",
    "conv.6.1": "BatchNorm_2",
    "conv.9.0": "QuantConv_3",
    "conv.9.1": "BatchNorm_3",
    "conv.11.conv": "QuantConv_4",
    "conv.11.conv_c": "QuantConv_5",
    "conv.13.0": "QuantConv_6",
    "conv.13.1": "BatchNorm_4",
    "conv.15.conv": "QuantConv_7",
    "conv.15.conv_c": "QuantConv_8",
    "fc.2.0": "QuantDense_0",
    "fc.5.0": "QuantDense_1",
}


def tcja_load_pretrained_weights(state, location):

  if ".pth" in location:

    torch_state = torch.load(location, map_location=torch.device("cpu"))

    torch_weights = unfreeze(
        jax.tree_util.tree_map(
            lambda x: x, state.params["params"]
        )
    )

    torch_bn_stats = unfreeze(
        jax.tree_util.tree_map(
            lambda x: x, state.batch_stats
        )
    )

    for key, value in torch_state["net"].items():

      if "num_batches_tracked" in key:
        continue

      map_key = ".".join(key.split(".")[:3])
      key_parts = key.split(".")

      if torch_map[map_key] is None:
        continue

      if "BatchNorm" in torch_map[map_key]:
        # batch norm params
        if key_parts[-1] == "weight":
          assert (
              torch_weights[torch_map[map_key]]["scale"].shape
              == value.shape
          )
          torch_weights[torch_map[map_key]]["scale"] = jnp.array(
              value
          )
          continue
        if key_parts[-1] == "bias":
          assert (
              torch_weights[torch_map[map_key]]["bias"].shape
              == value.shape
          )
          torch_weights[torch_map[map_key]]["bias"] = jnp.array(
              value
          )
          continue

        # batch stats
        if key_parts[-1] == "running_mean":
          assert (
              torch_bn_stats[torch_map[map_key]]["mean"].shape
              == value.shape
          )
          torch_bn_stats[torch_map[map_key]]["mean"] = jnp.array(
              value
          )
          continue
        if key_parts[-1] == "running_var":
          assert (
              torch_bn_stats[torch_map[map_key]]["var"].shape
              == value.shape
          )
          torch_bn_stats[torch_map[map_key]]["var"] = jnp.array(
              value
          )
          continue

      if ".weight" in key and "conv" in key:
        if len(value.shape) == 4:
          assert (
              torch_weights[torch_map[map_key]]["kernel"].shape
              == value.shape[::-1]
          )
          torch_weights[torch_map[map_key]][
              "kernel"
          ] = jnp.transpose(jnp.array(value), (2, 3, 1, 0))
          continue
        elif len(value.shape) == 3:
          assert (
              torch_weights[torch_map[map_key]]["kernel"].shape
              == value.shape[::-1]
          )
          # possibly wrong dimensionalty ordering here.
          torch_weights[torch_map[map_key]][
              "kernel"
          ] = jnp.transpose(jnp.array(value))
          continue
        else:
          raise Exception("Unknown weight dimensions...")

      if ".weight" in key and "fc" in key:
        assert (
            torch_weights[torch_map[map_key]]["kernel"].shape
            == value.shape[::-1]
        )
        torch_weights[torch_map[map_key]]["kernel"] = jnp.array(
            value
        ).transpose()
        continue

    general_params = {"params": torch_weights}
  else:

    chk_state = checkpoints.restore_checkpoint(location, None)
    chk_weights, _ = jax.tree_util.tree_flatten(
        chk_state["params"]["params"]
    )
    _, weight_def = jax.tree_util.tree_flatten(state.params["params"])
    params = jax.tree_util.tree_unflatten(weight_def, chk_weights)

    chk_batchstats, _ = jax.tree_util.tree_flatten(
        chk_state["batch_stats"]
    )
    _, batchstats_def = jax.tree_util.tree_flatten(state.batch_stats)
    torch_bn_stats = jax.tree_util.tree_unflatten(
        batchstats_def, chk_batchstats
    )

    general_params = {"params": params}

  return TrainState.create(
      apply_fn=state.apply_fn,
      batch_stats=freeze(torch_bn_stats),
      params=general_params,
      tx=state.tx,
  )
