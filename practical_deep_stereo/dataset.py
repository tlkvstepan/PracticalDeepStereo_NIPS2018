# Copyrights. All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.

import random

import cv2
import torch as th


class Dataset(object):
    def __init__(self, examples_files, transformers=None):
        """Returns initialized Dataset object.

        Args:
            examples_files: list of examples.
            transformers: list of the functions or objects with
                        "_call_" method, that takes example as
                        and input and return modified example.
        """
        self._examples_files = examples_files
        self._transformers = transformers

    def append_transformers(self, transformers):
        """Adds transformers to the dataset.

        Args:
            transformers: list of the functions or objects with
                        "_call_" method, that takes example as
                        and input and return modified example.
        """
        if not isinstance(transformers, list):
            raise ValueError('"transformers" should be a list.')
        if self._transformers is None:
            self._transformers = transformers
        else:
            self._transformers += transformers

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
        # OpenCv imread produces BGR image.
        image = cv2.cvtColor(cv2.imread(image_filename, 1), cv2.COLOR_BGR2RGB)
        image = th.from_numpy(image).float()
        return image.permute(2, 0, 1)

    def _read_disparity_image(self, example_files):
        """Returns disparity_image with indices [y, x].

        The locations with unknown disparity are set to infinity. If example
        does not come with the "disparity_image" the function returns None.
        """
        raise NotImplementedError(
            '"_read_disparity_image" method should be implemented in'
            'a child class.')

    def get_example(self, index):
        if index >= len(self):
            raise IndexError
        example_files = self._examples_files[index]
        example = {
            'left': {
                'image': self._read_image(example_files['left']['image']),
                'disparity_image': self._read_disparity_image(example_files)
            },
            'right': {
                'image': self._read_image(example_files['right']['image'])
            },
        }
        return example

    def __getitem__(self, index):
        """Returns example by its index.

        Returns:
            Dictionary that consists of: "left" and "right" items. In turn
            "left" item contain "image" and "disparity_image", whereas "right"
            contains only "image". The "image" is a 3D
            tensors, with indices [y, x, color_channel].
            The "disparity_image" is a 3D tensor, with indices [0, y, x]
            and values in range [0 ... disp_max] and unknown values set
            to "infinity". If example does not have the "disparity_image",
            the function returns only the "image".
        """
        example = self.get_example(index)
        if self._transformers is not None:
            for transformer in self._transformers:
                example = transformer(example)
        return example
