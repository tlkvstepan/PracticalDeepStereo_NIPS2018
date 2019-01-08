# Copyrights. All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.
import glob
import os
import random
import re

import numpy as np
import torch as th

from practical_deep_stereo import dataset

# The list below contains training examples with rendering artifacts.
# These images were found by visual inspection of images with largest
# training losses.
EXAMPLES_WITH_RENDERING_ARTIFACTS = [
    'TRAIN/B/0609/left/0011.png', 'TRAIN/B/0609/left/0010.png',
    'TRAIN/B/0653/left/0008.png', 'TRAIN/B/0653/left/0007.png',
    'TRAIN/B/0653/left/0009.png', 'TRAIN/B/0653/left/0010.png',
    'TRAIN/B/0653/left/0011.png', 'TRAIN/B/0653/left/0012.png',
    'TRAIN/B/0653/left/0006.png', 'TRAIN/C/0511/left/0006.png',
    'TRAIN/C/0511/left/0007.png', 'TRAIN/C/0511/left/0008.png',
    'TRAIN/C/0511/left/0009.png', 'TRAIN/C/0511/left/0010.png',
    'TRAIN/C/0511/left/0011.png', 'TRAIN/C/0511/left/0012.png',
    'TRAIN/C/0511/left/0013.png', 'TRAIN/C/0511/left/0014.png',
    'TRAIN/C/0511/left/0015.png', 'TRAIN/B/0386/left/0008.png',
    'TRAIN/B/0386/left/0009.png', 'TRAIN/B/0386/left/0010.png',
    'TRAIN/B/0386/left/0011.png', 'TRAIN/B/0386/left/0012.png',
    'TRAIN/B/0386/left/0013.png', 'TRAIN/B/0386/left/0014.png',
    'TRAIN/B/0386/left/0015.png', 'TRAIN/B/0576/left/0011.png',
    'TRAIN/B/0576/left/0012.png', 'TRAIN/B/0576/left/0013.png',
    'TRAIN/B/0576/left/0014.png', 'TRAIN/B/0576/left/0015.png',
    'TRAIN/B/0576/left/0008.png', 'TRAIN/B/0576/left/0009.png',
    'TRAIN/B/0576/left/0010.png', 'TRAIN/C/0599/left/0006.png',
    'TRAIN/C/0599/left/0007.png', 'TRAIN/C/0599/left/0008.png',
    'TRAIN/C/0599/left/0009.png', 'TRAIN/C/0599/left/0010.png',
    'TRAIN/C/0599/left/0011.png', 'TRAIN/C/0599/left/0012.png',
    'TRAIN/C/0599/left/0013.png', 'TRAIN/C/0599/left/0014.png',
    'TRAIN/C/0599/left/0015.png', 'TRAIN/A/0011/left/0012.png',
    'TRAIN/A/0011/left/0011.png', 'TRAIN/A/0011/left/0013.png',
    'TRAIN/A/0011/left/0014.png', 'TRAIN/A/0011/left/0015.png',
    'TRAIN/A/0534/left/0010.png', 'TRAIN/A/0534/left/0011.png',
    'TRAIN/A/0534/left/0012.png', 'TRAIN/A/0534/left/0013.png',
    'TRAIN/A/0690/left/0008.png', 'TRAIN/A/0690/left/0009.png',
    'TRAIN/A/0705/left/0008.png', 'TRAIN/A/0705/left/0009.png',
    'TRAIN/A/0705/left/0010.png', 'TRAIN/A/0705/left/0011.png',
    'TRAIN/A/0705/left/0012.png', 'TRAIN/A/0705/left/0013.png',
    'TRAIN/A/0705/left/0014.png', 'TRAIN/A/0705/left/0015.png',
    'TRAIN/B/0643/left/0006.png', 'TRAIN/B/0643/left/0007.png',
    'TRAIN/B/0643/left/0008.png', 'TRAIN/B/0643/left/0009.png',
    'TRAIN/B/0643/left/0010.png', 'TRAIN/B/0643/left/0011.png',
    'TRAIN/B/0643/left/0012.png', 'TRAIN/B/0643/left/0013.png',
    'TRAIN/B/0643/left/0014.png', 'TRAIN/B/0643/left/0015.png'
]
NUMBER_OF_VALIDATION_EXAMPLES = 500
MAXIMUM_DISPARITY_DURING_TRAINING = 255

# Following constant is used to mask large disparities, during
# evaluation according to "psm" protocol.
MAXIMUM_DISPARITY_DURING_TEST = 192

# Following constants are used to eliminate examples where many
# pixels have large disparities, during evaluation according to
# "crl" protocol.
MAXIMUM_PERCENTAGE_OF_LARGE_DISPARITIES = 25.0
LARGE_DISPARITY = 300.0
RECOMPUTE_DISPARITY_STATISTICS = False


def _read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None
    header = file.readline().decode("utf-8").rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("utf-8"))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')
    scale = float(file.readline().decode("utf-8").rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian
    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)
    data = np.ascontiguousarray(np.flipud(data))
    return data


def _filter_out_examples_with_large_disparities(examples):
    return [
        example for example in examples
        if (example['maximum_disparity'] <= MAXIMUM_DISPARITY_DURING_TRAINING
            and example['minimum_disparity'] >= 0)
    ]


def _filter_out_examples_with_too_many_large_disparities(examples):
    return [
        example for example in examples
        if not example['too_many_large_disparities']
    ]


def _is_example_with_artifacts(path_to_left_image):
    for example_with_artifacts in EXAMPLES_WITH_RENDERING_ARTIFACTS:
        if example_with_artifacts in path_to_left_image:
            return True
    return False


def _filter_out_examples_with_rendering_artifacts(examples):
    return [
        example for example in examples
        if not _is_example_with_artifacts(example['left_image'])
    ]


def _split_examples_into_training_and_test_sets(examples):
    test_examples = [
        example for example in examples if 'TEST' in example['left_image']
    ]
    training_examples = [
        example for example in examples if 'TRAIN' in example['left_image']
    ]
    return training_examples, test_examples


def _png_files_in_folder(folder):
    image_paths = glob.glob(os.path.join(folder, "*.png"))
    # To gaurantee same order of the files across different runs, sort
    # files in alphabetical order.
    return sorted([os.path.basename(image_path) for image_path in image_paths])


def _folders_with_left_images(images_folder):
    # For each directory "os.walk" returns list of tuples,
    # where each tuple consists of: path to the directory
    # (folder_structure[0]), list of sub-directories and list of files
    # in the directory (folder_structure[2]).
    folders_with_left_images = sorted([
        folder_structure[0] for folder_structure in os.walk(images_folder)
        if (len(folder_structure[2]) > 0 and 'left' in folder_structure[0])
    ])
    # To gaurantee same order of the directories across different
    # runs, sort directories in alphabetical order.
    return sorted(folders_with_left_images)


def _get_right_image_filename(left_image_file):
    basename = os.path.basename(left_image_file)
    parent_folder = os.path.abspath(
        os.path.join(os.path.dirname(left_image_file), os.pardir))
    return os.path.join(parent_folder, 'right', basename)


def _get_disparity_image_filename(left_image_file, images_folder,
                                  disparity_images_folder):
    basename = os.path.basename(left_image_file).split(".")[0] + '.pfm'
    folder = os.path.join(
        disparity_images_folder,
        os.path.dirname(os.path.relpath(left_image_file, images_folder)))
    return os.path.join(folder, basename)


def _get_disparity_statistics_filename(disparity_image_file):
    return disparity_image_file.split(".")[0] + '.txt'


def _is_too_many_large_disparities(disparity_image):
    return ((disparity_image > LARGE_DISPARITY).sum() * 100.0 / float(
        disparity_image.size)) > MAXIMUM_PERCENTAGE_OF_LARGE_DISPARITIES


def _compute_disparity_statistics(disparity_image_file):
    """Returns disparities statistics.

    If text file with pre-computed values exists, than the values are
    taken from this file, otherwise they are computed and save to the
    text file.

    Returns:
        disparity_minimum, disparitiy_maximum: boundaries of example's
                                               disparity range;
        too_many_large_disparities: "1" if more than 25% of pixels have
                                    disparity larger than 300 pixels
                                    and "0" otherwise.
    """
    disparity_range_file = _get_disparity_statistics_filename(
        disparity_image_file)
    if os.path.isfile(
            disparity_range_file) and not RECOMPUTE_DISPARITY_STATISTICS:
        (minimum_disparity, maximum_disparity,
         too_many_large_disparities) = \
            np.loadtxt(disparity_range_file, dtype=np.int, unpack=True)
    else:
        disparity_image = _read_pfm(disparity_image_file)
        minimum_disparity = int(np.floor(disparity_image.min()))
        maximum_disparity = int(np.ceil(disparity_image.max()))
        too_many_large_disparities = _is_too_many_large_disparities(
            disparity_image)
        np.savetxt(
            disparity_range_file,
            [minimum_disparity, maximum_disparity, too_many_large_disparities],
            fmt='%i')
    return minimum_disparity, maximum_disparity, too_many_large_disparities


def _find_examples(dataset_folder):
    """Returns list with FlyingThings3D dataset examples.

    Note, that the function returns examples in the same order across
    different runs.

    Args:
        dataset_folder: folder with FlyingThings3D dataset, that contains
                        "frames_cleanpass" folder with left and right
                        images and "disparity" folder with disparities.

    Returns:
        List of examples, where each example is a dictionary
        with following items:
        (1) "left_image" file with the left image;
        (2) "right_image" file with the right image ;
        (3) "disparity_image" file with the disparity image for the left
            camera;
        (4) "minimum_disparity" and "maximum_disparity" disparity range
            boundaries;
        (5) "too_many_large_disparities" flag that is set to "True" if
            more than 25% of pixels have disparity more than 300 pixels.
    """
    images_folder = os.path.join(dataset_folder, 'frames_cleanpass')
    disparity_images_folder = os.path.join(dataset_folder, 'disparity')
    folders_with_left_images = _folders_with_left_images(images_folder)
    examples = []
    for folder_index, folder_with_left_images in enumerate(
            folders_with_left_images):
        for basename in _png_files_in_folder(folder_with_left_images):
            left_image_file = os.path.join(folder_with_left_images, basename)
            right_image_file = _get_right_image_filename(left_image_file)
            disparity_image_file = _get_disparity_image_filename(
                left_image_file, images_folder, disparity_images_folder)
            (minimum_disparity, maximum_disparity, too_many_large_disparities
             ) = _compute_disparity_statistics(disparity_image_file)
            examples.append({
                'left_image':
                left_image_file,
                'right_image':
                right_image_file,
                'disparity_image':
                disparity_image_file,
                'minimum_disparity':
                minimum_disparity,
                'maximum_disparity':
                maximum_disparity,
                'too_many_large_disparities':
                too_many_large_disparities
            })
    return examples


def _mask_large_disparities(example):
    disparity_image = example['disparity_image']
    out_of_range_mask = ((disparity_image < 0) |
                         (disparity_image > MAXIMUM_DISPARITY_DURING_TEST))
    disparity_image[out_of_range_mask] = float('inf')
    return example


class FlyingThings3D(dataset.Dataset):
    """FlyingThings3D dataset."""

    def _read_disparity_image(self, example_files):
        disparity_image = _read_pfm(example_files['disparity_image'])
        return th.from_numpy(disparity_image).float()

    @classmethod
    def benchmark_dataset(cls, dataset_folder, is_psm_protocol):
        """Returns benchmark dataset.

        Args:
            dataset_folder: folder with FlyingThings3D dataset, that contains
                            "frames_cleanpass" folder with left and right
                            images and "disparity" folder with disparities.
            is_psm_protocol: If "True", the benchmarking dataset is generated
                             using "psm" protocol described in "Pyramid stereo
                             matching network" by Jia-Ren Chang et al,
                             otherwise it is generated using protocol described
                             in "Cascade Residual Learning: A Two-stage
                             Convolutional Neural Network for Stereo Matching"
                             by Jiahao Pang et al is used. According to the
                             "psm" protocol, pixels with ground truth
                             disparities larger than 192 pixels are masked out
                             and thus excluded from the evaluation. According
                             to the latter protocol, examples where more
                             than 25% of pixels have disparity more than 300
                             pixels are excluded from the evaluation.
        """
        examples = _find_examples(dataset_folder)
        examples = _split_examples_into_training_and_test_sets(examples)[1]
        if is_psm_protocol:
            transforms = [_mask_large_disparities]
            return FlyingThings3D(examples, transforms)
        examples = _filter_out_examples_with_too_many_large_disparities(
            examples)
        return FlyingThings3D(examples)

    @classmethod
    def training_split(cls, dataset_folder):
        """Returns training and validation datasets.

        Example from FlyingThings3d dataset is added to the training
        or validation datasets if:

        (1) it is training example of FlyingThings3d dataset;
        (2) it does not have rendering artifacts;
        (3) all its disparities are within the range [0, 255].

        The function always generates same split.

        Args:
            dataset_folder: folder with FlyingThings3D dataset, that contains
                            "frames_cleanpass" folder with left and right
                            images and "disparity" folder with disparities.
        """
        examples = _find_examples(dataset_folder)
        # Manual random seed garantees that splits will be same in a
        # different runs.
        random.seed(0)
        random.shuffle(examples)
        examples = _split_examples_into_training_and_test_sets(examples)[0]
        examples = _filter_out_examples_with_rendering_artifacts(examples)
        examples = _filter_out_examples_with_large_disparities(examples)
        validation_examples = examples[:NUMBER_OF_VALIDATION_EXAMPLES]
        training_examples = examples[NUMBER_OF_VALIDATION_EXAMPLES:]
        return FlyingThings3D(training_examples), FlyingThings3D(
            validation_examples)
