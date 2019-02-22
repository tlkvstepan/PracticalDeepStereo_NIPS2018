# Copyrights. All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.

import math

import torch as th

from practical_deep_stereo import errors


def test_compute_absolute_error():
    # yapf: disable
    estimated_disparity = th.Tensor([[1.0, 2.0],
                                     [3.0, 4.0]])
    ground_truth_disparity = th.Tensor([[2.0, 2.0],
                                        [float('inf'), 1.0]])
    # yapf: enable
    (pixelwise_absolute_error,
     mean_absolute_error) = errors.compute_absolute_error(
         estimated_disparity, ground_truth_disparity, use_mean=True)
    # yapf: disable
    assert th.all(
        th.isclose(pixelwise_absolute_error, th.Tensor([[1.0, 0.0],
                                                        [0.0, 3.0]])))
    # yapf: enable
    assert math.isclose(mean_absolute_error, 4.0 / 3.0, rel_tol=1e-3)
    (pixelwise_absolute_error,
     median_absolute_error) = errors.compute_absolute_error(
         estimated_disparity, ground_truth_disparity, use_mean=False)
    assert math.isclose(median_absolute_error, 1.0, rel_tol=1e-3)
    # yapf: disable
    assert th.all(
        th.isclose(pixelwise_absolute_error, th.Tensor([[1.0, 0.0],
                                                        [0.0, 3.0]])))
    # yapf: enable
    # Test case when no ground truth is avaliable.
    ground_truth_disparity = th.full((2, 2), float('inf'))
    (pixelwise_absolute_error,
     mean_absolute_error) = errors.compute_absolute_error(
         estimated_disparity, ground_truth_disparity, use_mean=True)
    assert mean_absolute_error == 0.0


def test_compute_n_pixels_error():
    # yapf: disable
    estimated_disparity = th.Tensor([[1.0, 2.0],
                                     [3.0, 4.0]])
    ground_truth_disparity = th.Tensor([[2.0, 2.0],
                                        [float('inf'), 1.0]])
    # yapf: enable
    (pixelwise_n_pixels_error, n_pixels_error) = errors.compute_n_pixels_error(
        estimated_disparity, ground_truth_disparity, n=1.0)
    # yapf: disable
    assert th.all(
        th.isclose(pixelwise_n_pixels_error, th.Tensor([[0.0, 0.0],
                                                        [0.0, 1.0]])))
    # yapf: enable
    assert math.isclose(n_pixels_error, 100.0 / 3.0, rel_tol=1e-3)
    # Test case when no ground truth is avaliable.
    ground_truth_disparity = th.full((2, 2), float('inf'))
    (pixelwise_n_pixels_error, n_pixels_error) = errors.compute_n_pixels_error(
        estimated_disparity, ground_truth_disparity, n=1.0)
    assert n_pixels_error == 0.0
