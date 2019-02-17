# Copyrights. All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.

import math

from torch import nn

from practical_deep_stereo import embedding
from practical_deep_stereo import estimator
from practical_deep_stereo import matching
from practical_deep_stereo import regularization


class _SizeAdapter(object):
    """Converts size of input to standard size.

    Practical deep network works only with input images
    which height and width are multiples of a minimum size.
    This class allows to pass to the network images of arbitrary
    size, by padding the input to the closest multiple
    and unpadding the network's output to the original size.
    """

    def __init__(self, minimum_size=64):
        self._minimum_size = minimum_size
        self._pixels_pad_to_width = None
        self._pixels_pad_to_height = None

    def _closest_larger_multiple_of_minimum_size(self, size):
        return int(math.ceil(size / self._minimum_size) * self._minimum_size)

    def pad(self, network_input):
        """Returns "network_input" paded with zeros to the "standard" size.

        The "standard" size correspond to the height and width that
        are closest multiples of "minimum_size". The method pads
        height and width  and and saves padded values. These
        values are then used by "unpad_output" method.
        """
        height, width = network_input.size()[-2:]
        self._pixels_pad_to_height = (
            self._closest_larger_multiple_of_minimum_size(height) - height)
        self._pixels_pad_to_width = (
            self._closest_larger_multiple_of_minimum_size(width) - width)
        return nn.ZeroPad2d((self._pixels_pad_to_width, 0,
                             self._pixels_pad_to_height, 0))(network_input)

    def unpad(self, network_output):
        """Returns "network_output" cropped to the original size.

        The cropping is performed using values save by the "pad_input"
        method.
        """
        return network_output[..., self._pixels_pad_to_height:, self.
                              _pixels_pad_to_width:]


class PdsNetwork(nn.Module):
    """Practical Deep Stereo (PDS) network."""

    def __init__(self, maximum_disparity=255):
        super(PdsNetwork, self).__init__()
        self._size_adapter = _SizeAdapter()
        self._embedding = embedding.Embedding()
        self._matching = matching.Matching(
            operation=matching.MatchingOperation(), maximum_disparity=0)
        self.set_maximum_disparity(maximum_disparity)
        self._regularization = regularization.Regularization()
        self._estimator = estimator.SubpixelMap()

    def set_maximum_disparity(self, maximum_disparity):
        """Reconfigure network for different disparity range."""
        if (maximum_disparity + 1) % 64 != 0:
            raise ValueError(
                '"maximum_disparity" + 1 should be multiple of 64, e.g.,'
                '"maximum disparity" can be equal to 63, 191, 255, 319...')
        self._maximum_disparity = maximum_disparity
        # During the embedding spatial dimensions of an input are downsampled
        # 4x times. Therefore, "maximum_disparity" of matching module is
        # computed as (maximum_disparity + 1) / 4 - 1.
        self._matching.set_maximum_disparity((maximum_disparity + 1) // 4 - 1)

    def pass_through_network(self, left_image, right_image):
        left_descriptor, shortcut_from_left = self._embedding(left_image)
        right_descriptor = self._embedding(right_image)[0]
        matching_signatures = self._matching(left_descriptor, right_descriptor)
        return self._regularization(matching_signatures,
                                    shortcut_from_left)

    def forward(self, left_image, right_image):
        """Returns sub-pixel disparity (or matching cost in training mode)."""
        network_output = self.pass_through_network(
            self._size_adapter.pad(left_image),
            self._size_adapter.pad(right_image))
        if not self.training:
            network_output = self._estimator(network_output)
        return self._size_adapter.unpad(network_output)
