# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer

"""Tests for DVS gesture input pipeline.

From "Incorporating Learnable Membrane Time Constant to Enhance Learning of
  Spiking Neural Networks" - https://arxiv.org/pdf/2007.05785.pdf

"""

from absl.testing import absltest

import jax
import numpy as np

import sys

sys.path.append("..")

from input_pipeline import preprocess_data_number  # noqa: E402
from bptt.configs import plif_dvsgesture as default_lib  # noqa: E402

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_disable_most_optimizations", True)


class dvsgestureNumberTest(absltest.TestCase):
  """Test cases for data preprocessing."""

  def test_dvsgesture_number(self):

    base_size = 128
    config = default_lib.get_config()

    out = preprocess_data_number(
        np.load("../data/plif_test/data_addrs.npy"),
        np.load("../data/plif_test/data_times.npy"),
        config,
        base_size,
    )

    orig_data = np.load("../data/plif_test/pre_proc_data.npy")

    assert (
        ~(np.transpose(orig_data, (0, 2, 3, 1)) == np.array(out))
    ).sum() == 0


if __name__ == "__main__":
  absltest.main()
