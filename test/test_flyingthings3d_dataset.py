# Copyrights. All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.

import pkg_resources

from practical_deep_stereo import flyingthings3d_dataset

FOLDER_WITH_FRAGMENT_OF_FLYINGTHINGS3D_DATASET = \
 pkg_resources.resource_filename(__name__, "data/flyingthings3d")


def _mockup_transform(example):
    return example


def _check_example_items(example):
    assert 'disparity_image' in example
    assert 'left_image' in example
    assert 'right_image' in example
    disparity_image = example['disparity_image']
    left_image = example['left_image']
    assert len(disparity_image.size()) == 2
    assert len(left_image.size()) == 3


def test_flyingthings3d_dataset():
    training_set, validation_set = \
     flyingthings3d_dataset.FlyingThings3D.training_split(
        FOLDER_WITH_FRAGMENT_OF_FLYINGTHINGS3D_DATASET,
        number_of_validation_examples=0,
        maximum_disparity=100)
    # Only one training example has all disparities < 100 pixels.
    assert (len(validation_set) == 0)
    assert (len(training_set) == 1)
    training_example = training_set[0]
    _check_example_items(training_example)
    assert training_example['disparity_image'].max() <= 100

    benchmark_set = flyingthings3d_dataset.FlyingThings3D.benchmark_dataset(
        FOLDER_WITH_FRAGMENT_OF_FLYINGTHINGS3D_DATASET,
        is_psm_protocol=True,
        maximum_disparity=63,
        maximum_percentage_of_large_disparities=10.0,
        large_disparity=80)
    benchmark_set.append_transforms([_mockup_transform])
    assert len(benchmark_set) == 2
    test_example = benchmark_set[0]
    disparity_image = test_example['disparity_image']
    assert disparity_image.max().item() == float('inf')
    assert disparity_image[disparity_image.ne(float('inf'))].max() <= 80.0

    benchmark_set = flyingthings3d_dataset.FlyingThings3D.benchmark_dataset(
        FOLDER_WITH_FRAGMENT_OF_FLYINGTHINGS3D_DATASET, is_psm_protocol=False,
        maximum_disparity=63,
        maximum_percentage_of_large_disparities=10.0,
        large_disparity=80)
    # Only one test examples has less that 10% of disparities larger than 80
    # pixels.
    assert len(benchmark_set) == 1
