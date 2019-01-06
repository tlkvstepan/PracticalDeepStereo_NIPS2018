# © All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.

from torch.nn import functional
import torch as th


class SubpixelMap(object):
    """Approximation of an sub-pixel MAP estimator.

    In every location (x, y), function collects similarity scores
    for disparities in a vicinty of a disparity with maximum similarity
    score and converts them to disparity distribution using softmax.
    Next, the disparity in every location (x, y) is computed as mean
    of this distribution.

    It is used only for inference.
    """

    def __init__(self, half_support_window=4, disparity_step=2):
        super(SubpixelMap, self).__init__()
        """Returns object of SubpixelMap class.

        Args:
            disparity_step: step in pixels between near-by disparities in
                            input "similarities" tensor.
            half_support_window: defines size of disparity window in pixels
                                 around disparity with maximum similarity,
                                 which is used to convert similarities
                                 to probabilities and compute mean.
        """
        if disparity_step < 1:
            raise ValueError('"disparity_step" should be positive integer.')
        if half_support_window < 1:
            raise ValueError(
                '"half_support_window" should be positive integer.')
        if half_support_window % disparity_step != 0:
            raise ValueError('"half_support_window" should be multiple of the'
                             '"disparity_step"')
        self._disparity_step = disparity_step
        self._half_support_window = half_support_window

    def __call__(self, similarities):
        """Returns sub-pixel disparity.

        Args:
            similarities: Tensor with similarities for every
                          disparity and every location with indices
                          [batch_index, disparity_index, y, x].

        Returns:
            Tensor with disparities for every location with
            indices [batch_index, y, x].
        """
        # In every location (x, y) find disparity with maximum similarity
        # score.
        maximum_similarity, disparity_index_with_maximum_similarity = \
            th.max(similarities, dim=1, keepdim=True)
        support_disparities, support_similarities = [], []
        maximum_disparity_index = similarities.size(1)

        # Collect similarity scores for the disparities around the disparity
        # with the maximum similarity score.
        for disparity_index_shift in range(
                -self._half_support_window // self._disparity_step,
                self._half_support_window // self._disparity_step + 1):
            disparity_index = (disparity_index_with_maximum_similarity +
                               disparity_index_shift).float()
            invalid_disparity_index_mask = (
                (disparity_index < 0) |
                (disparity_index >= maximum_disparity_index))
            disparity_index[invalid_disparity_index_mask] = 0
            nearby_similarities = th.gather(similarities, 1,
                                            disparity_index.long())
            nearby_similarities[invalid_disparity_index_mask] = -float('inf')
            support_similarities.append(nearby_similarities)
            nearby_disparities = th.gather(
                (self._disparity_step *
                 disparity_index).expand_as(similarities), 1,
                disparity_index.long())
            support_disparities.append(nearby_disparities)
        support_similarities = th.stack(support_similarities, dim=1)
        support_disparities = th.stack(support_disparities, dim=1)

        # Convert collected similarity scores to the disparity distribution
        # using softmax and compute disparity as a mean of this distribution.
        probabilities = functional.softmax(support_similarities, dim=1)
        disparities = th.sum(probabilities * support_disparities.float(), 1)
        return disparities.squeeze(1)
