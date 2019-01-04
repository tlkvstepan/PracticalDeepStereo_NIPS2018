# Copyrights. All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.

import random

import cv2
import torch as th


class Dataset(object):
    def __init__(self, examples_files, transforms=None):
        """Returns initialized Dataset object.

        Args:
            examples_files: list of examples.
            transforms: list of the functions or objects with
                        "_call_" method, that takes example as
                        and input and return modified example.
        """
        self._examples_files = examples_files
        self._transforms = transforms

    def append_transforms(self, transforms):
        """Adds transforms to the dataset.

        Args:
            transforms: list of the functions or objects with
                        "_call_" method, that takes example as
                        and input and return modified example.
        """
        if not isinstance(transforms, list):
            raise ValueError('"transforms" should be a list.')
        if self._transforms is None:
            self._transforms = transforms
        else:
            self._transforms += transforms

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
