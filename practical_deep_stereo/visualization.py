# Copyrights. All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.

import os

import torch as th
import numpy as np

# Matplotlib backend should be choosen before pyplot is imported.
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from mpl_toolkits import axes_grid1


def gray_to_color(array, colormap_name='jet', vmin=None, vmax=None):
    cmap = plt.get_cmap(colormap_name)
    norm = plt.Normalize(vmin, vmax)
    return cmap(norm(array))


def _add_scaled_colorbar(plot, aspect=20, pad_fraction=0.5, **kwargs):
    """Adds scaled colorbar to existing plot."""
    divider = axes_grid1.make_axes_locatable(plot.axes)
    width = axes_grid1.axes_size.AxesY(plot.axes, aspect=1. / aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_axis = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_axis)
    return plot.axes.figure.colorbar(plot, cax=cax, **kwargs)


def save_image(filename, image, color_first=True):
    """Save color image to file.

    Args:
        filename: image file where the image will be saved..
        image: 3d image tensor.
        color_first: if True, the color dimesion is the first
                     dimension of the "image", otherwise the
                     color dimesion is the last dimesion.
    """
    figure = plt.figure()
    if color_first:
        numpy_image = image.permute(1, 2, 0).numpy()
    else:
        numpy_image = image.numpy()
    plot = plt.imshow(numpy_image.astype(np.uint8))
    plot.axes.get_xaxis().set_visible(False)
    plot.axes.get_yaxis().set_visible(False)
    figure.savefig(filename, bbox_inches='tight', dpi=200)
    plt.close()


def save_matrix(filename,
                matrix,
                minimum_value=None,
                maximum_value=None,
                colormap='magma',
                is_colorbar=True):
    """Saves the matrix to the image file.

    Args:
        filename: image file where the matrix will be saved.
        matrix: tensor of size (height x width). Some values might be
                equal to inf.
        minimum_value, maximum value: boundaries of the range.
                                      Values outside ot the range are
                                      shown in white. The colors of other
                                      values are determined by the colormap.
                                      If maximum and minimum values are not
                                      given they are calculated as 0.001 and
                                      0.999 quantile.
        colormap: map that determines color coding of matrix values.
    """
    figure = plt.figure()
    noninf_mask = matrix != float('inf')
    if minimum_value is None:
        minimum_value = np.quantile(matrix[noninf_mask], 0.001)
    if maximum_value is None:
        maximum_value = np.quantile(matrix[noninf_mask], 0.999)
    plot = plt.imshow(
        matrix.numpy(), colormap, vmin=minimum_value, vmax=maximum_value)
    if is_colorbar:
        _add_scaled_colorbar(plot)
    plot.axes.get_xaxis().set_visible(False)
    plot.axes.get_yaxis().set_visible(False)
    figure.savefig(filename, bbox_inches='tight', dpi=200)
    plt.close()


def plot_points_on_background(points_coordinates,
                              background,
                              points_color=[0, 0, 255]):
    """
    Args:
        points_coordinates: array of (y, x) points coordinates
                            of size (number_of_points x 2).
        background: (3 x height x width)
                    gray or color image uint8.
        color: color of points [red, green, blue] uint8.
    """
    if not (len(background.size()) == 3 and background.size(0) == 3):
        raise ValueError('background should be (color x height x width).')
    _, height, width = background.size()
    background_with_points = background.clone()
    y, x = points_coordinates.transpose(0, 1)
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    if not (x_min >= 0 and y_min >= 0 and x_max < width and y_max < height):
        raise ValueError('points coordinates are outsize of "background" '
                         'boundries.')
    background_with_points[:, y, x] = th.Tensor(points_color).type_as(
        background).unsqueeze(-1)
    return background_with_points


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
    points_coordinates = th.nonzero(binary_error)
    washed_out_image = color_image // 2 + 128
    return plot_points_on_background(points_coordinates, washed_out_image)


class Logger(object):
    def __init__(self, filename):
        self._filename = filename

    def log(self, text):
        """Appends text line to the file."""
        if os.path.isfile(self._filename):
            handler = open(self._filename, 'r')
            lines = handler.readlines()
            handler.close()
        else:
            lines = []
        lines.append(text + '\n')
        handler = open(self._filename, 'w')
        handler.writelines(lines)
        handler.close()


def plot_losses_and_errors(filename,
                           losses,
                           errors,
                           righ_y_axis_label='Validation error, [%]'):
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
    error_axis.set_ylabel(righ_y_axis_label, color='red')
    error_axis.legend(handles=[loss_plot, error_plot])
    figure.savefig(filename, bbox_inches='tight')
    plt.close()
