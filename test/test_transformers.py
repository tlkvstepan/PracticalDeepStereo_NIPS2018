# Copyrights. All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.

import torch as th

from practical_deep_stereo import transformers


def test_central_crop():
    th.manual_seed(0)
    central_crop = transformers.CentralCrop(
        height=20,
        width=10,
        get_items_to_crop = (lambda x: [
            x['left']['image'], x['right']['image'],
            x['left']['disparity_image']]))
    example = {
        'left': {
            'image': th.rand(2, 3, 111, 302),
            'disparity_image': th.rand(2, 111, 302)
        },
        'right': {
            'image': th.rand(2, 3, 111, 302)
        }
    }
    example = central_crop(example)
    assert example['left']['image'].size() == (2, 3, 20, 10)
    assert example['left']['disparity_image'].size() == (2, 20, 10)
