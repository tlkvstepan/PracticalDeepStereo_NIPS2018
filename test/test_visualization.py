# Copyrights. All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.

import os
import tempfile

import torch as th

from practical_deep_stereo import visualization


def test_save_image():
    filename = tempfile.mkstemp(suffix='.png')[1]
    visualization.save_image(
        filename=filename, image=th.ones(3, 10, 20))
    assert os.path.isfile(filename)


def test_image_with_binary_error():
    binary_error = th.zeros(5, 7)
    binary_error[2, 3] = 1
    overlay = visualization.overlay_image_with_binary_error(
        color_image=th.full((3, 5, 7), 255).byte(),
        binary_error=binary_error.byte())
    assert overlay.size() == (3, 5, 7)
    assert th.all(overlay[:, 0, 0].squeeze() == th.ByteTensor([255, 255, 255]))
    assert th.all(overlay[:, 2, 3].squeeze() == th.ByteTensor([0, 0, 255]))


def test_save_matrix():
    filename = tempfile.mkstemp(suffix='.png')[1]
    visualization.save_matrix(
        filename=filename,
        matrix=th.rand(10, 20),
        minimum_value=0.1,
        maximum_value=0.9,
        colormap='magma')
    assert os.path.isfile(filename)


def test_plot_loss_and_error():
    filename = tempfile.mkstemp(suffix='.png')[1]
    visualization.plot_losses_and_errors(
        filename=filename,
        losses=[100.0, 60.0, 40.0, 10.0],
        errors=[3, 2, 1.9, 1.4])
    assert os.path.isfile(filename)
