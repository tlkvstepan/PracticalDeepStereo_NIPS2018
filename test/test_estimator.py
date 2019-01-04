# © All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.

import numpy as np

import torch as th
from torch import autograd

from practical_deep_stereo import estimator


def test_subpixel_map():
    subpixel_map_estimator = estimator.SubpixelMap(
        half_support_window=2, disparity_step=1)
    similarities = autograd.Variable(th.Tensor([0.1, 0.4, 0.3, 0.2,
                                                0.3])).view(1, 5, 1, 1)
    disparity = subpixel_map_estimator(similarities).squeeze().item()
    expected_disparity = 1.52
    assert np.isclose(expected_disparity, disparity, atol=1e-4)

    subpixel_map_estimator = estimator.SubpixelMap(
        half_support_window=2, disparity_step=2)
    disparity = subpixel_map_estimator(similarities).squeeze().item()
    expected_disparity = 2.124
    assert np.isclose(expected_disparity, disparity, atol=1e-4)
