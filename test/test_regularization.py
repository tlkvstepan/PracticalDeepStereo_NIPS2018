# Copirights. All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.

import torch as th

from practical_deep_stereo import regularization


def test_contraction_block_output_size():
    th.manual_seed(0)
    block_input = th.rand(2, 6, 10, 14, 16)
    contraction = regularization.ContractionBlock(number_of_features=6)
    downsampling_output, smoothing_output = contraction(block_input)
    assert downsampling_output.size() == (2, 12, 5, 7, 8)
    assert smoothing_output.size() == (2, 12, 5, 7, 8)


def test_expansion_block_output_size():
    th.manual_seed(0)
    block_input = th.rand(2, 6, 10, 14, 16)
    shortcut = th.rand(2, 3, 20, 28, 32)
    expansion = regularization.ExpansionBlock(number_of_features=6)
    block_output = expansion(block_input, shortcut)
    assert block_output.size() == (2, 3, 20, 28, 32)


def test_regularization_output_size():
    th.manual_seed(0)
    shortcut_from_left_image = th.rand(2, 8, 32, 32)
    matching_signatures = th.rand(2, 8, 32, 32, 32)
    regularization_module = regularization.Regularization()
    matching_cost = regularization_module(matching_signatures,
                                          shortcut_from_left_image)
    assert matching_cost.size() == (2, 64, 128, 128)
