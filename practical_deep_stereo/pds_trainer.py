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


def average(list_of_values):
    return th.Tensor(list_of_values).mean().item()


class PdsTrainer(trainer.Trainer):
    def _initialize_filenames(self):
        super(PdsTrainer, self)._initialize_filenames()
        self._left_image_template = os.path.join(self._experiment_folder,
                                                 'example_{0:04d}_image.png')
        self._estimated_disparity_image_template = os.path.join(
            self._experiment_folder,
            'example_{0:04d}_disparity_epoch_{1:03d}.png')
        self._ground_truth_disparity_image_template = os.path.join(
            self._experiment_folder,
            'example_{0:04d}_disparity_ground_truth.png')
        self._3_pixels_error_image_template = os.path.join(
            self._experiment_folder,
            'example_{0:04d}_error_map_epoch_{1:03d}.png')

    def _run_network(self, batch_or_example):
        batch_or_example['network_output'] = self._network(
            batch_or_example['left']['image'],
            batch_or_example['right']['image'])

    def _compute_gradients_wrt_loss(self, batch):
        # Note that "network_output" contains matching similarity.
        loss = self._criterion(batch['network_output'],
                               batch['left']['disparity_image'])
        loss.backward()
        batch['loss'] = loss.detach().item()
        del loss

    def _compute_error(self, example):
        # Note that the "network_output" contains estimated disparity image.
        binary_error_map, three_pixels_error = errors.compute_n_pixels_error(
            example['network_output'], example['left']['disparity_image'])
        mean_absolute_error = errors.compute_absolute_error(
            example['network_output'], example['left']['disparity_image'])[1]
        example['binary_error_map'] = binary_error_map
        example['error'] = {
            'three_pixels_error': three_pixels_error,
            'mean_absolute_error': mean_absolute_error
        }

    def _average_losses(self, losses):
        return average(losses)

    def _average_processing_time(self, processing_times):
        return average(processing_times)

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

    def _report_training_progress(self):
        """Plot and print training loss and validation error every epoch."""
        test_errors = list(
            map(lambda element: element['three_pixels_error'],
                self._test_errors))
        visualization.plot_losses_and_errors(
            self._plot_filename, self._training_losses, test_errors)
        self._logger.log('epoch {0:02d} ({1:02d}) : '
                         'training loss = {2:.5f}, '
                         'MAE = {3:.5f} [pix], '
                         '3PE = {4:.5f} [%], '
                         'learning rate = {5:.5f}.'.format(
                             self._current_epoch + 1, self._end_epoch,
                             self._training_losses[-1],
                             self._test_errors[-1]['mean_absolute_error'],
                             self._test_errors[-1]['three_pixels_error'],
                             trainer.get_learning_rate(self._optimizer)))

    def _visualize_example(self, example, example_index):
        """Visualizes validation examples.

        Saves estimated and ground truth disparity with similar scale,
        left image, and binary error map overlayed with the left image for
        3 examples.
        """
        if example_index <= self._number_of_examples_to_visualize:
            # Dataset loader adds additional singletone dimension at the
            # beggining of tensors.
            ground_truth_disparity_image = example['left']['disparity_image'][
                0].cpu()
            left_image = example['left']['image'][0].cpu().byte()
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
