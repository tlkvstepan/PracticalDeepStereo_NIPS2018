# Copyrights. All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.

import torch as th

from practical_deep_stereo import pds_network


def test_size_adapter():
    th.manual_seed(0)
    size_adapter = pds_network._SizeAdapter()
    tensor = th.rand(2, 3, 15, 67)
    padded_tensor = size_adapter.pad(tensor)
    assert padded_tensor.size() == (2, 3, 64, 128)
    unpadded_tensor = size_adapter.unpad(padded_tensor)
    assert unpadded_tensor.size() == (2, 3, 15, 67)


def test_pds_network():
    th.manual_seed(0)
    right_image = th.rand(1, 3, 62, 49)
    left_image = th.rand(1, 3, 62, 49)
    network = pds_network.PdsNetwork(63)

    network.train()
    matching_cost = network(left_image, right_image)
    assert matching_cost.size() == (1, 32, 62, 49)

    network.set_maximum_disparity(255)
    matching_cost = network(left_image, right_image)
    assert matching_cost.size() == (1, 128, 62, 49)

    network.eval()
    disparity = network(left_image, right_image)
    assert disparity.size() == (1, 62, 49)
