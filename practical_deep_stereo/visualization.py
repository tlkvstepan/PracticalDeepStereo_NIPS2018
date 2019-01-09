# Copyrights. All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.

import cv2
import torch as th

import matplotlib.pyplot as plt


def save_color_image(filename, color_image):
    """Save color image to file.

    Args:
        filename: image filename.
        color_image: color image tensor of size (color, height, width)
                     (RGB colors order).
    """
    # Note, that imwrite of OpenCv requires tensor:
    # (1) with BGR colors channels order;
    # (2) with dimensionality (height, width, color).
    cv2.imwrite(filename, cv2.cvtColor(
        color_image.permute([1, 2, 0]).numpy(), cv2.COLOR_RGB2BGR))

def save_matrix_as_image(matrix,
                         filename=None,
                         minimum_value=None,
                         maximum_value=None,
                         colormap='magma'):
    """Saves the matrix to the image file.

    Args:
        matrix: tensor of size (height x width).
        minimum_value, maximum value: boundaries of the range.
                                      Values outside ot the range are
                                      shown in white. The colors of other
                                      values are determined by the colormap.
        colormap: map that determines color coding of matrix values.
        filename: image file where the matrix is saved.
    """
    figure = plt.figure()
    if minimum_value is None:
        minimum_value = matrix.min()
    if maximum_value is None:
        maximum_value = matrix.max()
    plot = plt.imshow(
        matrix.numpy(), colormap, vmin=minimum_value, vmax=maximum_value)
    plot.axes.get_xaxis().set_visible(False)
    plot.axes.get_yaxis().set_visible(False)
    figure.savefig(filename, bbox_inches='tight', dpi=200)
    plt.close()


def overlay_image_with_binary_error(color_image, binary_error):
    """Returns byte image overlayed with the binary error.

    Contrast of the image is reduced, brightness is incrased,
    and locations with the errors are shown in blue.

    Args:
        color_image: byte image tensor of size
                     (color_index, height, width);
        binary_error: byte tensor of size (height x width),
                      where "True"s correspond to error,
                      and "False"s correspond to absence of error.
    """
    washed_out_image = color_image.float() * 0.5 + 128.0
    red, green, blue = th.chunk(washed_out_image, 3, dim=0)
    red, green, blue = red.squeeze(0), green.squeeze(0), blue.squeeze(0)
    red[binary_error], green[binary_error], blue[binary_error] = 0, 0, 255
    return th.stack([red, green, blue], 0).byte()


def plot_losses_and_errors(filename, losses, errors):
    """Plots the loss and the error.

    The plot has two y-axis: the left is reserved for the loss
    and the right is reserved for the error. The axis have
    different scale. The axis and the curve of the loss are shown
    in blue and the axis and the curve for the error are shown
    in red.

    Args:
        filename: image file where plot is saved;
        training_loss, validation_error: lists with loss and error values
                                         respectively. Every element of the
                                         list corresponds to an epoch.
    """
    epochs = range(1, len(losses) + 1)
    figure, loss_axis = plt.subplots()
    smallest_loss = min(losses)
    loss_label = 'Training loss (smallest {0:.3f})'.format(smallest_loss)
    loss_plot = loss_axis.plot(epochs, losses, 'bs-', label=loss_label)[0]
    loss_axis.set_ylabel('Training loss', color='blue')
    loss_axis.set_xlabel('Epoch')
    error_axis = loss_axis.twinx()
    smallest_error = min(errors)
    error_label = 'Validation error (smallest {0:.3f})'.format(smallest_error)
    error_plot = error_axis.plot(epochs, errors, 'ro--', label=error_label)[0]
    error_axis.set_ylabel('Validation error, [%]', color='red')
    error_axis.legend(handles=[loss_plot, error_plot])
    figure.savefig(filename, bbox_inches='tight')
    plt.close()
