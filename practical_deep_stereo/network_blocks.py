# © All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.

from torch import nn


def convolution_3x3x3(number_of_input_features, number_of_output_features,
                      stride):
    return nn.Conv3d(
        number_of_input_features,
        number_of_output_features,
        kernel_size=3,
        stride=stride,
        padding=1)


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


def transposed_convolution_3x4x4_stride_122(number_of_input_features,
                                            number_of_output_features):
    return nn.ConvTranspose3d(
        number_of_input_features,
        number_of_output_features,
        kernel_size=(3, 4, 4),
        stride=(1, 2, 2),
        padding=(1, 1, 1))


def convolution_block_2D_with_relu_and_instance_norm(number_of_input_features,
                                                     number_of_output_features,
                                                     kernel_size, stride):
    return nn.Sequential(
        nn.Conv2d(
            number_of_input_features,
            number_of_output_features,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
        nn.InstanceNorm2d(number_of_output_features, affine=True))


def convolution_block_3D_with_relu_and_instance_norm(number_of_input_features,
                                                     number_of_output_features,
                                                     kernel_size, stride):
    return nn.Sequential(
        nn.Conv3d(
            number_of_input_features,
            number_of_output_features,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
        nn.InstanceNorm3d(number_of_output_features, affine=True))


def transposed_convololution_block_3D_with_relu_and_instance_norm(
        number_of_input_features, number_of_output_features, kernel_size,
        stride, padding):
    return nn.Sequential(
        nn.ConvTranspose3d(
            number_of_input_features,
            number_of_output_features,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding), nn.LeakyReLU(negative_slope=0.1, inplace=True),
        nn.InstanceNorm3d(number_of_output_features, affine=True))


def convolutional_block_5x5_stride_2(number_of_input_features,
                                     number_of_output_features):
    return convolution_block_2D_with_relu_and_instance_norm(
        number_of_input_features,
        number_of_output_features,
        kernel_size=5,
        stride=2)


def convolutional_block_3x3(number_of_input_features,
                            number_of_output_features):
    return convolution_block_2D_with_relu_and_instance_norm(
        number_of_input_features,
        number_of_output_features,
        kernel_size=3,
        stride=1)


def convolutional_block_3x3x3(number_of_input_features,
                              number_of_output_features):
    return convolution_block_3D_with_relu_and_instance_norm(
        number_of_input_features,
        number_of_output_features,
        kernel_size=3,
        stride=1)


def convolutional_block_3x3x3_stride_2(number_of_input_features,
                                       number_of_output_features):
    return convolution_block_3D_with_relu_and_instance_norm(
        number_of_input_features,
        number_of_output_features,
        kernel_size=3,
        stride=2)


def transposed_convolutional_block_4x4x4_stride_2(number_of_input_features,
                                                  number_of_output_features):
    return transposed_convololution_block_3D_with_relu_and_instance_norm(
        number_of_input_features,
        number_of_output_features,
        kernel_size=4,
        stride=2,
        padding=1)


class ResidualBlock(nn.Module):
    """Residual block with nonlinearity before addition."""

    def __init__(self, number_of_features):
        super(ResidualBlock, self).__init__()
        self.convolutions = nn.Sequential(
            convolutional_block_3x3(number_of_features, number_of_features),
            convolutional_block_3x3(number_of_features, number_of_features))

    def forward(self, block_input):
        return self.convolutions(block_input) + block_input
