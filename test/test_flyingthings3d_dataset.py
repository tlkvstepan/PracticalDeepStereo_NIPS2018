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
    flyingthings3d_dataset.RECOMPUTE_DISPARITY_STATISTICS = True
    flyingthings3d_dataset.NUMBER_OF_VALIDATION_EXAMPLES = 0
    flyingthings3d_dataset.MAXIMUM_DISPARITY_DURING_TRAINING = 100
    flyingthings3d_dataset.MAXIMUM_DISPARITY_DURING_TEST = 32
    flyingthings3d_dataset.LARGE_DISPARITY = 80.0
    flyingthings3d_dataset.MAXIMUM_PERCENTAGE_OF_LARGE_DISPARITIES = 10.0

    training_set, validation_set = \
     flyingthings3d_dataset.FlyingThings3D.training_split(
        FOLDER_WITH_FRAGMENT_OF_FLYINGTHINGS3D_DATASET)
    # Only one training example has all disparities < 100 pixels.
    assert (len(validation_set) == 0)
    assert (len(training_set) == 1)
    training_example = training_set[0]
    _check_example_items(training_example)
    assert training_example['disparity_image'].max() <= 100

    benchmark_set = flyingthings3d_dataset.FlyingThings3D.benchmark_dataset(
        FOLDER_WITH_FRAGMENT_OF_FLYINGTHINGS3D_DATASET, is_psm_protocol=True)
    benchmark_set.append_transforms([_mockup_transform])
    assert len(benchmark_set) == 2
    test_example = benchmark_set[0]
    disparity_image = test_example['disparity_image']
    assert disparity_image.max().item() == float('inf')
    assert disparity_image[disparity_image.ne(float('inf'))].max() <= 80.0

    benchmark_set = flyingthings3d_dataset.FlyingThings3D.benchmark_dataset(
        FOLDER_WITH_FRAGMENT_OF_FLYINGTHINGS3D_DATASET, is_psm_protocol=False)
    # Only one test examples has less that 10% of disparities larger than 80
    # pixels.
    assert len(benchmark_set) == 1
