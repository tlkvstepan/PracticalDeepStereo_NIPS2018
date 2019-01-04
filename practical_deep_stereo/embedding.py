# Copyrights. All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.

from torch import nn


def _convolution_3x3(number_of_input_features, number_of_output_features):
    return nn.Conv2d(
        number_of_input_features,
        number_of_output_features,
        kernel_size=3,
        padding=1)


def _convolution_5x5_stride_2(number_of_input_features,
                              number_of_output_features):
    return nn.Conv2d(
        number_of_input_features,
        number_of_output_features,
        kernel_size=5,
        stride=2,
        padding=2)


def _convolutional_block_5x5_stride_2(number_of_input_features,
                                      number_of_output_features):
    return nn.Sequential(
        _convolution_5x5_stride_2(number_of_input_features,
                                  number_of_output_features),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
        nn.InstanceNorm2d(number_of_output_features, affine=True))


def _convolutional_block_3x3(number_of_input_features,
                             number_of_output_features):
    return nn.Sequential(
        _convolution_3x3(number_of_output_features, number_of_output_features),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
        nn.InstanceNorm2d(number_of_output_features, affine=True))


class _ResidualBlockWithPreactivation(nn.Module):
    """Residual block."""

    def __init__(self, number_of_features):
        super(_ResidualBlockWithPreactivation, self).__init__()
        self.convolutions = nn.Sequential(
            _convolutional_block_3x3(number_of_features, number_of_features),
            _convolutional_block_3x3(number_of_features, number_of_features))

    def forward(self, block_input):
        return self.convolutions(block_input) + block_input


class Embedding(nn.Module):
    """Embedding module."""

    def __init__(self,
                 number_of_input_features=3,
                 number_of_features=64,
                 number_of_residual_blocks=2):
        super(Embedding, self).__init__()
        embedding_modules = [
            _convolutional_block_5x5_stride_2(number_of_input_features,
                                              number_of_features),
            _convolutional_block_5x5_stride_2(number_of_features,
                                              number_of_features),
        ]
        embedding_modules += [
            _ResidualBlockWithPreactivation(number_of_features)
            for _ in range(number_of_residual_blocks)
        ]
        self._embedding_modules = nn.ModuleList(embedding_modules)

    def forward(self, image):
        """Returns image's descriptor.

        Args:
            image: color image of size
                   batch_size x 3 x height x width;
            descriptor: image's descriptor of size
                        batch_size x 64 x height / 4 x width / 4.
        """
        descriptor = image
        for embedding_module in self._embedding_modules:
            descriptor = embedding_module(descriptor)
        return descriptor
