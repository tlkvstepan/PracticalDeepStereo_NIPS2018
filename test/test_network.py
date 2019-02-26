# Copyrights. All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.

import torch as th

from practical_deep_stereo import network


def test_pds_network():
    th.manual_seed(0)
    right_image = th.rand(1, 3, 62, 49)
    left_image = th.rand(1, 3, 62, 49)
    pds_network = network.PdsNetwork.default(63)

    pds_network.train()
    matching_cost = pds_network(left_image, right_image)
    assert matching_cost.size() == (1, 32, 62, 49)

    pds_network.set_maximum_disparity(255)
    matching_cost = pds_network(left_image, right_image)
    assert matching_cost.size() == (1, 128, 62, 49)

    pds_network.eval()
    disparity = pds_network(left_image, right_image)
    assert disparity.size() == (1, 62, 49)
