# Copyrights. All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.


class CentralCrop(object):
    """Cropes same central area from left, right and disparity image."""

    def __init__(self, crop_height, crop_width):
        self._crop_height = crop_height
        self._crop_width = crop_width

    def __call__(self, example):
        (left_image, right_image,
         disparity_image) = (example['left_image'], example['right_image'],
                             example['disparity_image'])
        height, width = left_image.size()[-2:]
        x_start = (width - self._crop_width) // 2
        y_start = (height - self._crop_height) // 2
        x_end = x_start + self._crop_width
        y_end = y_start + self._crop_height

        (example['left_image'], example['right_image'],
         example['disparity_image']) = (
             left_image[..., y_start:y_end, x_start:x_end].clone(),
             right_image[..., y_start:y_end, x_start:x_end].clone(),
             disparity_image[..., y_start:y_end, x_start:x_end].clone())

        return example
