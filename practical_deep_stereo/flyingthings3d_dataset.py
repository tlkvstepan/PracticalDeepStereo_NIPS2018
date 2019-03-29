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


def _filter_out_examples_with_large_disparities(examples, maximum_disparity):
    return [
        example for example in examples
        if (example['maximum_disparity'] <= maximum_disparity
            and example['minimum_disparity'] >= 0)
    ]


def _filter_out_examples_with_too_many_large_disparities(
        examples, maximum_percentage_of_large_disparities, large_disparity):
    return [
        example for example in examples
        if (100.0 - example['cumulative_distribution_from_0_to_511']
            [large_disparity]) < maximum_percentage_of_large_disparities
    ]


def _is_example_with_artifacts(path_to_left_image):
    for example_with_artifacts in EXAMPLES_WITH_RENDERING_ARTIFACTS:
        if example_with_artifacts in path_to_left_image:
            return True
    return False


def _filter_out_examples_with_rendering_artifacts(examples):
    return [
        example for example in examples
        if not _is_example_with_artifacts(example['left']['image'])
    ]


def _split_examples_into_training_and_test_sets(examples):
    test_examples = [
        example for example in examples if 'TEST' in example['left']['image']
    ]
    training_examples = [
        example for example in examples if 'TRAIN' in example['left']['image']
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


def _get_disparity_statistic_filename(disparity_image_file):
    return disparity_image_file.split(".")[0] + '.npz'


def _read_disparity_statistic(disparity_statistic_file):
    file_content = np.load(disparity_statistic_file)
    return (int(file_content['minimum_disparity']),
            int(file_content['maximum_disparity']),
            file_content['cumulative_distribution'])


def _compute_cumulative_distribution(disparity_image, minimum_disparity,
                                     maximum_disparity):
    bins = [min(minimum_disparity, 0)] + list(range(
        1, 512)) + [max(maximum_disparity, 512)]
    histogram = np.histogram(disparity_image[:], bins=bins)[0]
    histogram = histogram / histogram.sum()
    return np.cumsum(histogram) * 100.0


def _compute_and_save_disparity_statistic(disparity_image_file,
                                          disparity_statistic_file):
    """Computes and saves disparity statistic.

    Computes integer minimum disparity, maximum disparity and descrete
    cumulative distribution from 0 to 255 (out-of-range disparities are
    added to the boundaries). In resulting cumulative distribution array
    n-th element contains percentage of pixels with disparity less
    or equal to n.

    Args:
        disparity_image_filename: pfm file with disparity image.
        disparity_statistic_file: numpy file where statistic is saved.
    """
    disparity_image = _read_pfm(disparity_image_file)
    minimum_disparity = int(np.floor(disparity_image.min()))
    maximum_disparity = int(np.ceil(disparity_image.max()))
    cumulative_distribution = _compute_cumulative_distribution(
        disparity_image, minimum_disparity, maximum_disparity)
    np.savez(
        disparity_statistic_file,
        minimum_disparity=minimum_disparity,
        maximum_disparity=maximum_disparity,
        cumulative_distribution=cumulative_distribution)


def _find_examples(dataset_folder):
    """Returns list with FlyingThings3D dataset examples.

    Note, that the function returns examples in the same order across
    different runs.

    Args:
        dataset_folder: folder with FlyingThings3D dataset, that contains
                        "frames_cleanpass" folder with left and right
                        images and "disparity" folder with disparities.

    Returns:
        List of examples, where each example is a dictionary with following
        items:
        (1) "left" with the "image" and "disparity_image" items;
        (2) "right" with the "image" item;
        (4) "minimum_disparity" and "maximum_disparity" disparity range
            boundaries;
        (5) "cumulative_distribution_from_0_to_511" cumulative distribution
            of disparities from 0 to 511. Out of range disparities
            contribute to the boundary bins.
    """
    dataset_folder = os.path.abspath(dataset_folder)
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
            disparity_statistic_file = _get_disparity_statistic_filename(
                disparity_image_file)
            if not os.path.isfile(disparity_statistic_file):
                _compute_and_save_disparity_statistic(
                    disparity_image_file, disparity_statistic_file)
            (minimum_disparity, maximum_disparity,
             cumulative_distribution_from_0_to_511
             ) = _read_disparity_statistic(disparity_statistic_file)
            examples.append({
                'left': {
                    'image': left_image_file,
                    'disparity_image': disparity_image_file
                },
                'right': {
                    'image': right_image_file
                },
                'minimum_disparity':
                minimum_disparity,
                'maximum_disparity':
                maximum_disparity,
                'cumulative_distribution_from_0_to_511':
                cumulative_distribution_from_0_to_511
            })
    return examples


def _mask_large_disparities(example, maximum_disparity):
    disparity_image = example['left']['disparity_image']
    out_of_range_mask = ((disparity_image < 0) |
                         (disparity_image > maximum_disparity))
    disparity_image[out_of_range_mask] = float('inf')
    return example


class FlyingThings3D(dataset.Dataset):
    """FlyingThings3D dataset."""

    def _read_disparity_image(self, example_files):
        disparity_image = _read_pfm(example_files['left']['disparity_image'])
        return th.from_numpy(disparity_image).float()

    @classmethod
    def benchmark_dataset(cls,
                          dataset_folder,
                          is_psm_protocol,
                          maximum_disparity=192,
                          maximum_percentage_of_large_disparities=25.0,
                          large_disparity=300):
        """Returns benchmark dataset.

        One of two benchmarking protocols can be used: "psm" or "crl". "psm"
        protocol is described in "Pyramid stereo matching network" by Jia-Ren
        Chang et al. The "crl" protocol is described in "Cascade Residual
        Learning: A Two-stage Convolutional Neural Network for Stereo Matching"
        by Jiahao Pang. According to the "crl" protocol examples where more
        than "maximum_percentage_of_large_disparities"=25% of pixels have
        disparity larger than "large_disparity"=300 pixels are excluded
        from the evaluation. Note, that according to both protocols pixels
        with ground truth disparity larger than maximum_disparity=192 are
        excluded from evaluation, since network this is a largest disparity
        that network can produce.

        Args:
            dataset_folder: folder with FlyingThings3D dataset, that contains
                            "frames_cleanpass" folder with left and right
                            images and "disparity" folder with disparities.
            is_psm_protocol: if "True", the "psm" protocol is used, otherwise
                             "crl" protocol is used.
            maximum_disparity: parameter of "psm" protocol.
            maximum_percentage_of_large_disparities, large_disparities:
                    parameter of "clr" protocol.
        """
        examples = _find_examples(dataset_folder)
        examples = _split_examples_into_training_and_test_sets(examples)[1]
        transformers = [
            lambda input: _mask_large_disparities(input, maximum_disparity)
        ]
        if is_psm_protocol:
            return FlyingThings3D(examples, transformers)
        examples = _filter_out_examples_with_too_many_large_disparities(
            examples, maximum_percentage_of_large_disparities, large_disparity)
        return FlyingThings3D(examples, transformers)

    @classmethod
    def training_split(cls,
                       dataset_folder,
                       number_of_validation_examples=500,
                       maximum_disparity=255):
        """Returns training and validation datasets.

        Example from FlyingThings3d dataset is added to the training
        or validation datasets if:

        (1) it is training example of FlyingThings3d dataset;
        (2) it does not have rendering artifacts;
        (3) all its disparities are within the range [0, maximum_disparity].

        Args:
            dataset_folder: folder with FlyingThings3D dataset, that contains
                            "frames_cleanpass" folder with left and right
                            images and "disparity" folder with disparities.
            number_of_validation_examples: number of examples from training set
                            that will be used for validation.
            maximum_disparity: maximum disparity in training / validation
                            dataset. All training examples with disparity
                            larger than "maximum_disparity" are excluded
                            from the dataset.
        """
        examples = _find_examples(dataset_folder)
        # Manual random seed garantees that splits will be same in a
        # different runs.
        random.seed(0)
        random.shuffle(examples)
        examples = _split_examples_into_training_and_test_sets(examples)[0]
        examples = _filter_out_examples_with_rendering_artifacts(examples)
        examples = _filter_out_examples_with_large_disparities(
            examples, maximum_disparity)
        _dataset = FlyingThings3D(examples)
        validation_dataset, training_dataset = _dataset.split_in_two(
            size_of_first_subset=number_of_validation_examples)
        return training_dataset, validation_dataset
