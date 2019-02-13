# Copyrights. All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.

from collections import defaultdict
import os

import torch as th

from practical_deep_stereo import errors
from practical_deep_stereo import visualization
from practical_deep_stereo import trainer


class PdsTrainer(trainer.Trainer):
    def _initialize_filenames(self):
        super(PdsTrainer, self)._initialize_filenames()
        self._left_image_template = os.path.join(self._experiment_folder,
                                                 'example_{0:02d}_image.png')
        self._estimated_disparity_image_template = os.path.join(
            self._experiment_folder,
            'example_{0:02d}_disparity_epoch_{1:03d}.png')
        self._ground_truth_disparity_image_template = os.path.join(
            self._experiment_folder,
            'example_{0:02d}_disparity_ground_truth.png')
        self._3_pixels_error_image_template = os.path.join(
            self._experiment_folder,
            'example_{0:02d}_error_map_epoch_{1:03d}.png')

    def _run_network(self, batch_or_example):
        batch_or_example['network_output'] = self._network(
            batch_or_example['left_image'], batch_or_example['right_image'])

    def _compute_loss(self, batch):
        # Note that "network_output" contains matching similarity.
        batch['loss'] = self._criterion(batch['network_output'],
                                        batch['disparity_image'])

    def _compute_error(self, example):
        # Note that the "network_output" contains estimated disparity image.
        binary_error_map, three_pixels_error = errors.compute_n_pixels_error(
            example['network_output'], example['disparity_image'])
        mean_absolute_error = errors.compute_absolute_error(
            example['network_output'], example['disparity_image'])[1]
        example['binary_error_map'] = binary_error_map
        example['error'] = {
            'three_pixels_error': three_pixels_error,
            'mean_absolute_error': mean_absolute_error
        }

    def _average_errors(self, errors):
        average_errors = defaultdict(lambda: [])
        for example_error in errors:
            for error_name, error_value in example_error.items():
                average_errors[error_name].append(error_value)
        return {
            error_name: th.Tensor(error_list).mean().item()
            for error_name, error_list in average_errors.items()
        }

    def _report_test_results(self, error, time):
        self._logger.log('Testing results:'
                         'MAE = {0:.5f} [pix], '
                         '3PE = {1:.5f} [%], '
                         'time-per-image = {2:.2f} [sec].'.format(
                             error['mean_absolute_error'],
                             error['three_pixels_error'], time))

    def _average_losses(self, losses):
        return th.Tensor(losses).mean().item()

    def _average_processing_time(self, processing_times):
        return th.Tensor(processing_times).mean().item()

    def _report_training_progress(self):
        """Plot and print training loss and validation error every epoch."""
        validation_errors = list(
            map(lambda element: element['three_pixels_error'],
                self._validation_errors))
        visualization.plot_losses_and_errors(
            self._plot_filename, self._training_losses, validation_errors)
        self._logger.log(
            'epoch {0:02d} ({1:02d}) : '
            'training loss = {2:.5f}, '
            'MAE = {3:.5f} [pix], '
            '3PE = {4:.5f} [%], '
            'learning rate = {5:.5f}.'.format(
                self._current_epoch + 1, self._end_epoch,
                self._training_losses[-1],
                self._validation_errors[-1]['mean_absolute_error'],
                self._validation_errors[-1]['three_pixels_error'],
                trainer.get_learning_rate(self._optimizer)))

    def _visualize_example(self, example, example_index):
        """Save visualization for examples.

        For the visualization, in addition to "disparity_image", "left_image",
        and "network_output" (that contains estimated disparity) the example
        should contain "binary_error_map".
        """
        if example_index <= 3:
            # Dataset loader adds additional singletone dimension at the
            # beggining of tensors.
            ground_truth_disparity_image = example['disparity_image'][0].cpu()
            left_image = example['left_image'][0].cpu().byte()
            estimated_disparity_image = example['network_output'][0].cpu()
            binary_error_map = example['binary_error_map'][0].cpu().byte()
            visualization.save_image(
                filename=self._left_image_template.format(example_index + 1),
                image=left_image)
            # Ensures same scale of the ground truth and estimated disparity.
            noninf_mask = ~th.isinf(ground_truth_disparity_image)
            minimum_disparity = ground_truth_disparity_image.min()
            maximum_disparity = ground_truth_disparity_image[noninf_mask].max()
            visualization.save_matrix(
                filename=self._ground_truth_disparity_image_template.format(
                    example_index + 1),
                matrix=ground_truth_disparity_image,
                minimum_value=minimum_disparity,
                maximum_value=maximum_disparity)
            visualization.save_matrix(
                filename=self._estimated_disparity_image_template.format(
                    example_index + 1, self._current_epoch + 1),
                matrix=estimated_disparity_image,
                minimum_value=minimum_disparity,
                maximum_value=maximum_disparity)
            image_overlayed_with_errors =\
                            visualization.overlay_image_with_binary_error(
                left_image, binary_error_map)
            visualization.save_image(
                filename=self._3_pixels_error_image_template.format(
                    example_index + 1, self._current_epoch + 1),
                image=image_overlayed_with_errors)
