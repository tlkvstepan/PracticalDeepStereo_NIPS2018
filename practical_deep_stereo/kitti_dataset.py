# Copyrights. All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.

import os
import random

import cv2
import torch as th

from practical_deep_stereo import dataset


def _find_examples(left_images_folder,
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
        List of examples, where each example is a dictionary with following
        items:
        (1) "left" with "image", "disparity_image" and
        "reflective_disparity_image" items. If the disparity
            is not avaliable last two items are set to None;
        (2) "right" with "image" item.
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
            'left': {
                'image': left_image_file,
                'disparity_image': disparity_image_file,
                'reflective_disparity_image': reflective_disparity_image_file
            },
            'right': {
                'image': right_image_file
            }
        })
        example_index += 1

    return examples


class Kitti(dataset.Dataset):
    """Kitti dataset.

    This is a combination of Kitti2012 and Kitti2015 datatasets.
    Ground truth disparity for occluded and non-occluded areas is used
    during training. For Kitti2012 dataset ground truth disparities
    for reflective and non-reflective areas are combined.

    Note that maximum disparity in the dataset is 231 pixels.
    """

    def _read_disparity_image(self, example_files):
        disparity_image_file = example_files['left']['disparity_image']
        reflective_disparity_image_file = example_files['left'][
            'reflective_disparity_image']
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
        return disparity_image

    @classmethod
    def training_split(cls, dataset_folder, number_of_validation_examples=58):
        """Returns training and validation datasets.

        The function always generates same split. For validation
        the function allocates 58 examples.

        Args:
            dataset_folder: folder that contains "data_stereo_flow"
                    folder with Kitti2012 dataset and "data_scene_flow"
                    folder with Kitti2015 dataset.
            number_of_validation_examples: number of examples from the training
                    set that are used for validation.
        """
        examples = _find_examples(
            left_images_folder=os.path.join(dataset_folder, 'data_stereo_flow',
                                            'training', 'colored_0'),
            right_images_folder=os.path.join(
                dataset_folder, 'data_stereo_flow', 'training', 'colored_1'),
            disparity_images_folder=os.path.join(
                dataset_folder, 'data_stereo_flow', 'training', 'disp_occ'),
            reflective_disparity_images_folder=os.path.join(
                dataset_folder, 'data_stereo_flow', 'training',
                'disp_refl_occ'))

        examples += _find_examples(
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

        return Kitti(training_examples), Kitti(validation_examples)

    @classmethod
    def kitti2015_benchmark(cls, dataset_folder):
        """Returns Kitti2015 benchmark dataset.

        Args:
            dataset_folder: folder that contains "data_scene_flow"
                      folder with Kitti2015 dataset.
        """
        examples = _find_examples(
            left_images_folder=os.path.join(dataset_folder, 'data_scene_flow',
                                            'testing', 'image_2'),
            right_images_folder=os.path.join(dataset_folder, 'data_scene_flow',
                                             'testing', 'image_3'))
        return Kitti(examples)

    @classmethod
    def kitti2012_benchmark(cls, dataset_folder):
        """Returns Kitti2012 benchmark dataset.

        Args:
            dataset_folder: folder that contains "data_stereo_flow"
                      folder with Kitti2012 dataset.
        """
        examples = _find_examples(
            left_images_folder=os.path.join(dataset_folder, 'data_stereo_flow',
                                            'testing', 'colored_0'),
            right_images_folder=os.path.join(
                dataset_folder, 'data_stereo_flow', 'testing', 'colored_1'))
        return Kitti(examples)
