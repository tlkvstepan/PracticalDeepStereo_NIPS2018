# Copyrights. All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.
import torch as th

from practical_deep_stereo import size_adapter


def test_size_adapter():
    adapter = size_adapter.SizeAdapter()
    input = th.rand(1, 10, 63, 100)
    padded = adapter.pad(input)
    assert padded.size() == (1, 10, 64, 128)
    unpadded = adapter.unpad(padded)
    assert (unpadded == input).all()
