# Copyrights. All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.

import torch as th

from practical_deep_stereo import embedding


def test_embedding_output_size():
    embedding_module = embedding.Embedding()
    th.manual_seed(0)
    image = th.rand(2, 3, 100, 100)
    descriptor = embedding_module(image)
    assert descriptor.size() == (2, 64, 25, 25)
