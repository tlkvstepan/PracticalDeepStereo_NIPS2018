# Copyrights. All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.

import os
import random

import cv2
import torch as th


class Dataset(object):
    def __init__(self, examples_files, transforms=None):
        self._examples_files = examples_files
        self._transforms = transforms

    def subsample(self, number_of_examples, random_seed=None):
        """Keeps "number_of_examples" examples in the dataset.

        By setting "random_seed", one can ensure that subset of examples
        will be same in a different runs. This method is usefull for
        debugging.
        """
        if random_seed is not None:
            random.seed(random_seed)
        self._examples_files = random.sample(self._examples_files,
                                             number_of_examples)

    def __len__(self):
        return len(self._examples_files)

    def _read_image(self, image_filename):
        """Returns image with indices [color_channel, y, x]."""
        image = th.from_numpy(cv2.imread(image_filename, 1)).float()
        return image.permute(2, 0, 1)

    def _read_disparity_image(self, example_files):
        """Returns disparity_image with indices [0, y, x].

        The locations with unknown disparity are set to infinity. If example
        does not come with the "disparity_image" the function returns None.
        """
        raise NotImplementedError(
            '"_read_disparity_image" method should be implemented in'
            'a child class.')

    def __getitem__(self, index):
        """Returns example by its index.

        Returns:
            Dictionary that consists of: "left_image", "right_image",
            "disparity_image". The "left_image" and "right_image" are 3D
            tensors, with indices [y, x, color_channel].
            The "disparity_image" is a 3D tensor, with indices [0, y, x]
            and values in range [0 ... disp_max] and unknown values set
            to "infinity". If example does not have the "disparity_image",
            the function returns only the "left_image" and the "right_image".
        """
        if index >= len(self):
            raise IndexError
        example_files = self._examples_files[index]
        example = {
            'left_image': self._read_image(example_files['left_image']),
            'right_image': self._read_image(example_files['right_image']),
            'disparity_image': self._read_disparity_image(example_files)
        }
        if self._transforms is not None:
            for transform in self._transforms:
                example = transform(example)
        return example


def _kitti_read_disparity(disparity_image_file,
                          reflective_disparity_image_file):
    if disparity_image_file is None:
        return None
    disparity_image = th.from_numpy(cv2.imread(disparity_image_file,
                                               0)).float()
    if reflective_disparity_image_file is not None:
        reflective_disparity_image = th.from_numpy(
            cv2.imread(reflective_disparity_image_file, 0)).float()
        reflective_disparity_image = reflective_disparity_image
        reflective_disparity_avaliable = reflective_disparity_image.ne(0)
        disparity_image[reflective_disparity_avaliable] = \
            reflective_disparity_image[reflective_disparity_avaliable]
    # Unknown disparities correspond to "0"s in the "disparity_image".
    disparity_unavaliable = disparity_image.eq(0)
    disparity_image[disparity_unavaliable] = float('inf')
    return disparity_image.unsqueeze(0)


def _kitti_find_examples(left_images_folder,
                         right_images_folder,
                         disparity_images_folder=None,
                         reflective_disparity_images_folder=None):
    """Returns list with Kitti dataset examples.

    Note, the function returns examples in same order across different
    runs.

    Args:
        left_images_folder: folder with left color images;
        right_images_folder: folder with right color images;
        disparity_images_folder: with sparse disparity images for
                                the left camera (specify only if avaliable);
        reflective_images_folder:  folder with reflective disparity
                                images (specify only if avaliable).

    Returns:
        List of examples, where each example consists of:
        (1) "left_image" file with the left image;
        (2) "right_image" file with the right image;
        (3) "disparity_image" file with the sparse disparity image
            for the left camera (if the disparity image is not avaliable,
            this item is set to None);
        (4) "reflective_disparity_image" file with the reflective
            disparity image for the left camera (if the disparity image is
            not avaliable, this item is set to None).
    """
    examples = []
    example_index = 0
    while True:
        basename = '{0:06d}_10.png'.format(example_index)
        left_image_file = os.path.join(left_images_folder, basename)
        if not os.path.isfile(left_image_file):
            break
        right_image_file = os.path.join(right_images_folder, basename)
        disparity_image_file = None
        reflective_disparity_image_file = None
        if disparity_images_folder is not None:
            disparity_image_file = os.path.join(disparity_images_folder,
                                                basename)
        if reflective_disparity_images_folder is not None:
            reflective_disparity_image_file = os.path.join(
                reflective_disparity_images_folder, basename)
        examples.append({
            'left_image':
            left_image_file,
            'right_image':
            right_image_file,
            'disparity_image':
            disparity_image_file,
            'reflective_disparity_image':
            reflective_disparity_image_file
        })
        example_index += 1

    return examples


class KittiDataset(Dataset):
    """Kitti dataset.

    This is a combination of Kitti2012 and Kitti2015 datatasets.
    Ground truth disparity for occluded and non-occluded areas is used
    during training. For Kitti2012 dataset ground truth disparities
    for reflective and non-reflective areas are combined.

    Note that maximum disparity in the dataset is 231 pixels.
    """

    def _read_disparity_image(self, example_files):
        return _kitti_read_disparity(
            example_files['disparity_image'],
            example_files['reflective_disparity_image'])

    @classmethod
    def training_split(cls, dataset_folder, number_of_validation_examples=58):
        """Returns training and validation datasets.

        Note, the splits are the same across the runs if
        "number_of_validation_examples" is the same.

        Args:
            dataset_folder: folder that contains "data_stereo_flow"
                    folder with Kitti2012 dataset and "data_scene_flow"
                    folder with Kitti2015 dataset.
            number_of_validation_examples: number of examples that will be
                      used for validation.
        """
        examples = _kitti_find_examples(
            left_images_folder=os.path.join(dataset_folder, 'data_stereo_flow',
                                            'training', 'colored_0'),
            right_images_folder=os.path.join(
                dataset_folder, 'data_stereo_flow', 'training', 'colored_1'),
            disparity_images_folder=os.path.join(
                dataset_folder, 'data_stereo_flow', 'training', 'disp_occ'),
            reflective_disparity_images_folder=os.path.join(
                dataset_folder, 'data_stereo_flow', 'training',
                'disp_refl_occ'))

        examples += _kitti_find_examples(
            left_images_folder=os.path.join(dataset_folder, 'data_scene_flow',
                                            'training', 'image_2'),
            right_images_folder=os.path.join(dataset_folder, 'data_scene_flow',
                                             'training', 'image_3'),
            disparity_images_folder=os.path.join(
                dataset_folder, 'data_scene_flow', 'training/disp_occ_0'),
            reflective_disparity_images_folder=None)

        # This garantee that splits will be same in a different runs.
        random.seed(0)
        random.shuffle(examples)
        validation_examples = examples[0:number_of_validation_examples]
        training_examples = examples[number_of_validation_examples:]

        return KittiDataset(training_examples), KittiDataset(
            validation_examples)

    @classmethod
    def kitti2015_benchmark_datasetset(cls, dataset_folder):
        """Returns Kitti2015 benchmark dataset.

        Args:
            dataset_folder: folder that contains "data_scene_flow"
                      folder with Kitti2015 dataset.
        """

        examples = _kitti_find_examples(
            left_images_folder=os.path.join(dataset_folder, 'data_scene_flow',
                                            'testing', 'image_2'),
            right_images_folder=os.path.join(dataset_folder, 'data_scene_flow',
                                             'testing', 'image_3'))
        return KittiDataset(examples)

    @classmethod
    def kitti2012_benchmark_datasetset(cls, dataset_folder):
        """Returns Kitti2012 benchmark dataset.

        Args:
            dataset_folder: folder that contains "data_stereo_flow"
                      folder with Kitti2012 dataset.
        """

        examples = _kitti_find_examples(
            left_images_folder=os.path.join(dataset_folder, 'data_stereo_flow',
                                            'testing', 'colored_0'),
            right_images_folder=os.path.join(
                dataset_folder, 'data_stereo_flow', 'testing', 'colored_1'))
        return KittiDataset(examples)
