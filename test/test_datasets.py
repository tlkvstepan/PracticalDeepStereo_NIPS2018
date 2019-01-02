# Copyrights. All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.

import pkg_resources

from practical_deep_stereo import datasets

KITTI_ROOTPATH = pkg_resources.resource_filename(__name__, "data/kitti")


def _mockup_transform(example):
    return example


def _check_fields(example):
    assert 'disparity_image' in example
    assert 'left_image' in example
    assert 'right_image' in example


def _check_training_example(example):
    _check_fields(example)
    disparity_image, left_image = example['left_image'], example[
        'disparity_image']
    assert len(disparity_image.size()) == 3
    assert len(left_image.size()) == 3


def _check_test_example(example):
    _check_fields(example)
    disparity_image = example['disparity_image']
    assert disparity_image is None


def test_kitti_dataset():
    training_set, validation_set = datasets.KittiDataset.training_split(
        KITTI_ROOTPATH, number_of_validation_examples=2)
    training_set.transforms = [_mockup_transform]
    assert len(validation_set) == 2

    example = validation_set[0]
    _check_training_example(example)

    validation_set.subsample(number_of_examples=1)
    assert len(validation_set) == 1

    kitti2012_benchmark_set = \
     datasets.KittiDataset.kitti2012_benchmark_datasetset(KITTI_ROOTPATH)
    len(kitti2012_benchmark_set) == 2
    example = kitti2012_benchmark_set[1]
    _check_test_example(example)
