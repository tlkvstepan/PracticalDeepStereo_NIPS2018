# © All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.

import numpy as np
import torch as th

from torch import autograd

from practical_deep_stereo import loss


def test_subpixel_cross_entropy():
    similarities = autograd.Variable(
        th.Tensor([[0.1, 0.3, 0.2, 0.05], [0.2, 0.1, 0.4, 0.0]]).view(
            2, 4, 1, 1))
    ground_truth_disparity = autograd.Variable(
        th.Tensor([[1.3], [float('inf')]]).view(2, 1, 1))
    criterion = loss.SubpixelCrossEntropy(
        diversity=2.0, disparity_step=1)
    expected_cross_entropy = 1.3654
    assert np.isclose(
        criterion(similarities, ground_truth_disparity).item(),
        expected_cross_entropy,
        atol=1e-3)
