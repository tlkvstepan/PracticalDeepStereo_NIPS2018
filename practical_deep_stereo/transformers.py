# Copyrights. All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.


class CentralCrop(object):
    """Cropes same central area from left, right and disparity image."""

    def __init__(self, height, width, get_items_to_crop):
        """Returns initialized transformer.

        Args:
            height, width: size of central crop;
            get_items_to_crop: function that given example, returns
                               list of items that should be cropped.
        """
        self._height = height
        self._width = width
        self._get_items_to_crop = get_items_to_crop

    def __call__(self, example):
        items_to_crop = self._get_items_to_crop(example)
        height, width = items_to_crop[0].size()[-2:]
        x_start = (width - self._width) // 2
        y_start = (height - self._height) // 2
        x_end = x_start + self._width
        y_end = y_start + self._height

        for index, item_to_crop in enumerate(items_to_crop):
            # Change data assigned to the reference.
            items_to_crop[index].data = item_to_crop[..., y_start:y_end,
                                                     x_start:x_end].data

        return example
