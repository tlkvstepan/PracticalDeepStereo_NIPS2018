# © All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.

from torch import nn


def convolution_3x3(number_of_input_features, number_of_output_features):
    return nn.Conv2d(
        number_of_input_features,
        number_of_output_features,
        kernel_size=3,
        padding=1)


def convolution_5x5_stride_2(number_of_input_features,
                             number_of_output_features):
    return nn.Conv2d(
        number_of_input_features,
        number_of_output_features,
        kernel_size=5,
        stride=2,
        padding=2)


def convolutional_block_5x5_stride_2(number_of_input_features,
                                     number_of_output_features):
    return nn.Sequential(
        convolution_5x5_stride_2(number_of_input_features,
                                 number_of_output_features),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
        nn.InstanceNorm2d(number_of_output_features, affine=True))


def convolutional_block_3x3(number_of_input_features,
                            number_of_output_features):
    return nn.Sequential(
        convolution_3x3(number_of_input_features, number_of_output_features),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
        nn.InstanceNorm2d(number_of_output_features, affine=True))


class ResidualBlock(nn.Module):
    """Residual block with nonlinearity before addition."""

    def __init__(self, number_of_features):
        super(ResidualBlock, self).__init__()
        self.convolutions = nn.Sequential(
            convolutional_block_3x3(number_of_features, number_of_features),
            convolutional_block_3x3(number_of_features, number_of_features))

    def forward(self, block_input):
        return self.convolutions(block_input) + block_input
