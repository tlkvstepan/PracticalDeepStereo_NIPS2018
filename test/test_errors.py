# Copyrights. All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.

import math

import torch as th

from practical_deep_stereo import errors


def _check_compute_absolute_error(estimated_disparity, ground_truth_disparity):
    (pixelwise_absolute_error,
     mean_absolute_error) = errors.compute_absolute_error(
         estimated_disparity, ground_truth_disparity)
    assert th.all(
        th.isclose(pixelwise_absolute_error, th.Tensor([[1.0, 0.0], [0.0,
                                                                     3.0]])))
    assert math.isclose(mean_absolute_error, 4.0 / 3.0, rel_tol=1e-3)


def _check_compute_n_pixels_error(estimated_disparity, ground_truth_disparity):
    (pixelwise_n_pixels_error, n_pixels_error) = errors.compute_n_pixels_error(
        estimated_disparity, ground_truth_disparity, n=1.0)
    assert th.all(
        th.isclose(pixelwise_n_pixels_error, th.Tensor([[0.0, 0.0], [0.0,
                                                                     1.0]])))
    assert math.isclose(n_pixels_error, 100.0 / 3.0, rel_tol=1e-3)


def test_errors():
    estimated_disparity = th.Tensor([[1.0, 2.0],
                                     [3.0, 4.0]])   # yapf: disable
    ground_truth_disparity = th.Tensor([[2.0, 2.0],
                                        [float('inf'), 1.0]])  # yapf: disable
    _check_compute_absolute_error(estimated_disparity, ground_truth_disparity)
    _check_compute_n_pixels_error(estimated_disparity, ground_truth_disparity)
