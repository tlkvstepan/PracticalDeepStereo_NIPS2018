# Copyrights. All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.

import torch as th

from practical_deep_stereo import transforms


def test_central_crop():
    th.manual_seed(0)
    central_crop = transforms.CentralCrop(crop_height=20, crop_width=10)
    example = {
        'left_image': th.rand(2, 3, 111, 302),
        'right_image': th.rand(2, 3, 111, 302),
        'disparity_image': th.rand(2, 111, 302)
    }
    example = central_crop(example)
    assert example['left_image'].size() == (2, 3, 20, 10)
    assert example['disparity_image'].size() == (2, 20, 10)
