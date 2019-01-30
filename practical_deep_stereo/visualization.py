# Copyrights. All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.

import os

import torch as th

# Matplotlib backend should be choosen before pyplot is imported.
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from mpl_toolkits import axes_grid1


def _add_scaled_colorbar(plot, aspect=20, pad_fraction=0.5, **kwargs):
    """Adds scaled colorbar to existing plot."""
    divider = axes_grid1.make_axes_locatable(plot.axes)
    width = axes_grid1.axes_size.AxesY(plot.axes, aspect=1. / aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_axis = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_axis)
    return plot.axes.figure.colorbar(plot, cax=cax, **kwargs)


def save_image(filename, image):
    """Save color image to file.

    Args:
        filename: image file where the image will be saved..
        color_image: color image tensor of size (color, height, width)
                     (RGB colors order).
    """
    figure = plt.figure()
    plot = plt.imshow(image.permute(1, 2, 0).numpy())
    plot.axes.get_xaxis().set_visible(False)
    plot.axes.get_yaxis().set_visible(False)
    figure.savefig(filename, bbox_inches='tight', dpi=200)
    plt.close()


def save_matrix(filename,
                matrix,
                minimum_value=None,
                maximum_value=None,
                colormap='magma'):
    """Saves the matrix to the image file.

    Args:
        filename: image file where the matrix will be saved.
        matrix: tensor of size (height x width). Some values might be
                equal to inf.
        minimum_value, maximum value: boundaries of the range.
                                      Values outside ot the range are
                                      shown in white. The colors of other
                                      values are determined by the colormap.
        colormap: map that determines color coding of matrix values.
    """
    figure = plt.figure()
    if minimum_value is None:
        minimum_value = matrix.min()
    if maximum_value is None:
        maximum_value = matrix[~th.isinf(minimum_value)].max()
    plot = plt.imshow(
        matrix.numpy(), colormap, vmin=minimum_value, vmax=maximum_value)
    _add_scaled_colorbar(plot)
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


class Logger(object):
    """Logger with line overwriting capability.

    Line overwriting can increase readability of the
    log file by minimizing number of lines in the file.
    """

    def __init__(self, filename):
        self._filename = filename

    def log(self, text, overwrite_line=False):
        """Appends text line to the file.

        Args:
            text: text line.
            overwrite_line: if flag is True when
                            last line in the file
                            is substituted by new.
        """
        if os.path.isfile(self._filename):
            handler = open(self._filename, 'r')
            lines = handler.readlines()
            if overwrite_line:
                lines = lines[:-1]
            handler.close()
        else:
            lines = []
        lines.append(text)
        handler = open(self._filename, 'w')
        handler.writelines(lines)
        handler.close()


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
