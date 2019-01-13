# Copirights. All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.

from torch import nn

from practical_deep_stereo import network_blocks


class ContractionBlock(nn.Module):
    """Contraction block, that downsamples the input.

    The contraction blocks constitute the contraction part of
    the regularization network. Each block consists of 2x
    "donwsampling" convolution followed by conventional "smoothing"
    convolution.
    """

    def __init__(self, number_of_features):
        super(ContractionBlock, self).__init__()
        self._downsampling_2x = network_blocks.convolutional_block_3x3x3_stride_2(
            number_of_features, 2 * number_of_features)
        self._smoothing = network_blocks.convolutional_block_3x3x3(
            2 * number_of_features, 2 * number_of_features)

    def forward(self, block_input):
        output_of_downsampling_2x = self._downsampling_2x(block_input)
        return output_of_downsampling_2x, self._smoothing(
            output_of_downsampling_2x)


class ExpansionBlock(nn.Module):
    """Expansion block, that upsamples the input.

    The expansion blocks constitute the expansion part of
    the regularization network. Each block consists of 2x
    "upsampling" transposed convolution and
    conventional "smoothing" convolution. The output of the
    "upsampling" convolution is summed with the
    "shortcut_from_contraction" and is fed to the "smoothing"
    convolution.
    """

    def __init__(self, number_of_features):
        super(ExpansionBlock, self).__init__()
        self._upsampling_2x = \
            network_blocks.transposed_convolutional_block_4x4x4_stride_2(
                    number_of_features, number_of_features // 2)
        self._smoothing = network_blocks.convolutional_block_3x3x3(
            number_of_features // 2, number_of_features // 2)

    def forward(self, block_input, shortcut_from_contraction):
        output_of_upsampling = self._upsampling_2x(block_input)
        return self._smoothing(output_of_upsampling +
                               shortcut_from_contraction)


class Regularization(nn.Module):
    """Regularization module, that enforce stereo matching constraints.

    It is a hourglass 3D convolutional network that consists
    of contraction and expansion parts, with the shortcut connections
    between them.

    The network downsamples the input 16x times along the spatial
    and disparity dimensions and then upsamples it 64x times along
    the spatial dimensions and 32x times along the disparity
    dimension, effectively computing matching cost only for even
    disparities.
    """

    def __init__(self, number_of_features=8):
        """Returns initialized regularization module."""
        super(Regularization, self).__init__()
        self._smoothing = network_blocks.convolutional_block_3x3x3(
            number_of_features, number_of_features)
        self._contraction_blocks = nn.ModuleList([
            ContractionBlock(number_of_features * scale)
            for scale in [1, 2, 4, 8]
        ])
        self._expansion_blocks = nn.ModuleList([
            ExpansionBlock(number_of_features * scale)
            for scale in [16, 8, 4, 2]
        ])
        self._upsample_to_halfsize = \
            network_blocks.transposed_convolutional_block_4x4x4_stride_2(
                number_of_features, number_of_features // 2)
        self._upsample_to_fullsize = \
            network_blocks.transposed_convolution_3x4x4_stride_122(
                number_of_features // 2, 1)

    def forward(self, matching_signatures, shortcut_from_left_image):
        """Returns regularized matching cost tensor.

        Args:
            matching_signatures: concatenated compact matching signatures
                                 for every disparity. It is tensor of size
                                 (batch_size, number_of_features,
                                 maximum_disparity / 4, height / 4,
                                 width / 4).
            shortcut_from_left_image: shortcut connection from the left
                                 image descriptor. It has size of
                                 (batch_size, number_of_features, height / 4,
                                  width / 4);

        Returns:
            regularized matching cost tensor of size (batch_size,
            maximum_disparity / 2, height, width). Every element of this
            tensor along the disparity dimension is a matching cost for
            disparity 0, 2, .. , maximum_disparity.
        """
        shortcuts_from_contraction = []
        shortcut = shortcut_from_left_image.unsqueeze(2)
        output = self._smoothing(matching_signatures)
        for contraction_block in self._contraction_blocks:
            shortcuts_from_contraction.append(output)
            shortcut, output = contraction_block(shortcut + output)

        del shortcut
        for expansion_block in self._expansion_blocks:
            output = expansion_block(output, shortcuts_from_contraction.pop())

        return self._upsample_to_fullsize(
            self._upsample_to_halfsize(output)).squeeze_(1)
