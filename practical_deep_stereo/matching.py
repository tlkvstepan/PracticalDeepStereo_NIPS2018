# © All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.

import torch as th
from torch import nn


class Matching(nn.Module):
    def __init__(self, maximum_disparity, operation):
        """Returns matching module.

        Args:
            maximum_disparity: Upper limit of disparity range
                               [0, maximum_disparity].
            operation: Operation that is applied to concatenated
                       left-right discriptors for all disparities.
                       This can be network module of function.
        """
        super(Matching, self).__init__()
        self._maximum_disparity = maximum_disparity
        self._pad = nn.ZeroPad2d((maximum_disparity, 0, 0, 0))
        self._operation = operation

    def set_disparity(self, maximum_disparity):
        """Change disparity range."""
        self._maximum_disparity = maximum_disparity

    def forward(self, left_embedding, right_embedding):
        """Return result of forward pass.

        Args:
            left_embedding, right_embedding: Tensors for left and right
                            image embeddings with indices
                            [batch_index, feature_index, y, x].

        Returns:
            matching_signature: 4D tensor that contains concatenated
                                matching signatures (or matching score,
                                depending on "operation") for every disparity.
                                Tensor has indices
                                [batch_index, feature_index, disparity_index, y, x].
        """
        # Pad zeros from the left to the right embedding
        padded_right_embedding = self._pad(right_embedding)
        matching_signatures = []
        concatenated_embedding = th.cat(
            [left_embedding, right_embedding], dim=1)
        matching_signatures.append(self._operation(concatenated_embedding))
        for disparity in range(1, self._maximum_disparity + 1):
            shifted_right_embedding = padded_right_embedding[:, :, :, self.
                                                             _maximum_disparity
                                                             - disparity:
                                                             -disparity]
            concatenated_embedding = th.cat(
                [left_embedding, shifted_right_embedding], dim=1)
            matching_signatures.append(self._operation(concatenated_embedding))
        return th.stack(matching_signatures, dim=2)
