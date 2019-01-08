# Copyrights. All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.

import torch as th


def compute_absolute_error(estimated_disparity, ground_truth_disparity):
    """Returns pixel-wise and mean absolute error.

    Locations where ground truth is not avaliable do not contribute to mean
    absolute error. In such locations pixel-wise error is shown as zero.

    Args:
        ground_truth_disparity: ground truth disparity where locations with
                                unknow disparity are set to inf's.
        estimated_disparity: estimated disparity.
    """
    absolute_difference = (estimated_disparity - ground_truth_disparity).abs()
    locations_without_ground_truth = th.isinf(ground_truth_disparity)
    pixelwise_absolute_error = absolute_difference.clone()
    pixelwise_absolute_error[locations_without_ground_truth] = 0
    mean_absolute_error = \
            absolute_difference[~locations_without_ground_truth].mean().item()
    return pixelwise_absolute_error, mean_absolute_error


def compute_n_pixels_error(estimated_disparity, ground_truth_disparity, n=3.0):
    """Return pixel-wise n-pixels error and % of pixels with n-pixels error.

    Locations where ground truth is not avaliable do not contribute to mean
    n-pixel error. In such locations pixel-wise error is shown as zero.

    Note that n-pixel error is equal to one if
    |estimated_disparity-ground_truth_disparity| > 1 and zero otherwise.

    Args:
        ground_truth_disparity: ground truth disparity where locations with
                                unknow disparity are set to inf's.
        estimated_disparity: estimated disparity.
        n: maximum absolute disparity difference, that does not trigger
           n-pixel error.
    """
    locations_without_ground_truth = th.isinf(ground_truth_disparity)
    more_than_n_pixels_absolute_difference = (
        estimated_disparity - ground_truth_disparity).abs().gt(n).float()
    pixelwise_n_pixels_error = more_than_n_pixels_absolute_difference.clone()
    pixelwise_n_pixels_error[locations_without_ground_truth] = 0.0
    percentage_of_pixels_with_error = more_than_n_pixels_absolute_difference[
        ~locations_without_ground_truth].mean().item() * 100
    return pixelwise_n_pixels_error, percentage_of_pixels_with_error
