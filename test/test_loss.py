# © All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.

import numpy as np
import torch as th

from practical_deep_stereo import loss


def test_subpixel_cross_entropy():
    # yapf: disable
    similarities = th.Tensor([[0.1, 0.3, 0.2, 0.05],
                              [0.2, 0.1, 0.4, 0.0]])
    expected_similarities_gradient = th.Tensor([
        [0.0262, -0.0567, -0.0219, 0.0524],
        [0.0,     0.0,     0.0,    0.0]])
    # yapf: enable
    expected_similarities_gradient = th.transpose(
        expected_similarities_gradient, 0, 1).view(1, 4, 2, 1)
    similarities = th.transpose(similarities, 0, 1).view(1, 4, 2, 1)
    similarities.requires_grad_(True)
    ground_truth_disparity = th.Tensor([[1.3], [float('inf')]]).view(1, 2, 1)
    criterion = loss.SubpixelCrossEntropy(diversity=2.0, disparity_step=1)
    cross_entropy = criterion(similarities, ground_truth_disparity)
    expected_cross_entropy = 1.3654
    cross_entropy.backward()
    assert np.isclose(cross_entropy.item(), expected_cross_entropy, atol=1e-3)
    assert th.all(
        th.isclose(
            similarities.grad, expected_similarities_gradient, atol=1e-3))
