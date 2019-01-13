# Copyrights. All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.

import logging
import os

import torch as th

from practical_deep_stereo import errors
from practical_deep_stereo import visualization


def _is_on_cuda(network):
    return next(network.parameters()).is_cuda


def _is_logging_required(example_index, number_of_examples):
    """Returns True only if logging is required.

    Logging is performed for 10th, 20th, ... 100th quantiles.
    """
    return (example_index + 1) % max(1, number_of_examples // 10) == 0


def _move_tensors_to_cuda(dictionary_of_tensors):
    if isinstance(dictionary_of_tensors, dict):
        return {
            key: _move_tensors_to_cuda(value)
            for key, value in dictionary_of_tensors.items()
        }
    return dictionary_of_tensors.cuda()


def _change_logging_file(filename):
    file_handler = logging.FileHandler(filename, 'a')
    rootlog = logging.getLogger()
    for old_file_handler in rootlog.handlers[:]:
        rootlog.removeHandler(old_file_handler)
    rootlog.addHandler(file_handler)


class _Trainer(object):
    def __init__(self, parameters):
        """Returns initialized trainer object.

        Args:
            parameters: dictionary with parameters, that
                        should have the same names as
                        parameters of the object (without
                        underscore).
        """
        self._experiment_folder = None
        self._current_epoch = 0
        self._end_epoch = None
        self._learning_rate_scheduler = None
        self._network = None
        self._training_set_loader = None
        self._validation_set_loader = None
        self._optimizer = None
        self._criterion = None
        self._training_losses = []
        self._validation_errors = []
        self._from_dictionary(parameters)

    def _from_dictionary(self, parameters):
        attributes = vars(self)
        for key, value in parameters.items():
            _key = '_{0}'.format(key)
            attributes[_key] = value

    def _define_files(self):
        self._log_filename = os.path.join(self._experiment_folder, 'log.txt')
        self._plot_filename = os.path.join(self._experiment_folder, 'plot.png')
        self._checkpoint_template = os.path.join(self._experiment_folder,
                                                 '{0:03d}_checkpoint.bin')

    def load_checkpoint(self, filename, load_only_network=False):
        """Initilizes trainer from checkpoint.

        Args:
            filename: file with the checkpoint.
            load_only_network: if the flag is set, the function only loads
                               the network (can be usefull for
                               fine-tuning).
        """
        checkpoint = th.load(filename)
        if load_only_network:
            self._network.load_state_dict(checkpoint['network'])
            return
        parameters = {
            'current_epoch': len(checkpoint['training_losses']),
            'training_losses': checkpoint['training_losses'],
            'validation_errors': checkpoint['validation_errors']
        }
        self._from_dictionary(parameters)
        self._optimizer.load_state_dict(checkpoint['optimizer'])
        self._learning_rate_scheduler.load_state_dict(
            checkpoint['learning_rate_scheduler'])

    def _save_checkpoint(self):
        th.save({
            'training_losses':
            self._training_losses,
            'validation_errors':
            self._validation_errors,
            'network':
            self._network.state_dict(),
            'optimizer':
            self._optimizer.state_dict(),
            'learning_rate_scheduler':
            self._learning_rate_scheduler.state_dict()
        }, self._checkpoint_template.format(self._current_epoch + 1))

    def train(self):
        """Train network."""
        if th.cuda.is_available():
            th.backends.cudnn.fastest = True
            th.backends.cudnn.benchmark = True
        self._define_files()
        logging.basicConfig(
            filename=self._log_filename,
            format='%(asctime)s : %(message)s',
            datefmt="%m-%d %H:%M",
            level=logging.INFO)
        # basicConfig does not change file handler if it
        # was already assigned, therefore we have to change the file handler
        # manually.
        _change_logging_file(self._log_filename)
        start_epoch = self._current_epoch
        if start_epoch == self._end_epoch:
            return None
        for self._current_epoch in range(start_epoch, self._end_epoch):
            training_losses = self._train_for_epoch()
            validation_errors = self._validate()
            epoch_training_loss = th.Tensor(training_losses).mean()
            epoch_validation_error = th.Tensor(validation_errors).mean()
            self._training_losses.append(epoch_training_loss)
            self._validation_errors.append(epoch_validation_error)
            self._learning_rate_scheduler.step()
            visualization.plot_losses_and_errors(self._plot_filename,
                                                 self._training_losses,
                                                 self._validation_errors)
            self._save_checkpoint()
            logging.info('epoch {0:02d} ({1:02d}) : '
                         'training loss = {2:.5f}, '
                         'validation error = {3:.5f}, '
                         'learning rate = {4:.5f}.'.format(
                             self._current_epoch + 1, self._end_epoch,
                             epoch_training_loss, epoch_validation_error,
                             self._learning_rate_scheduler.get_lr()[0]))
        self._current_epoch = self._end_epoch
        return epoch_validation_error

    def _run_network(self, batch):
        """Computes network output and adds it to "batch"."""
        raise NotImplementedError('"_run_network" method should '
                                  'be implemented in a child class.')

    def _compute_loss(self, batch):
        """Computes training loss and adds it to "batch" as "loss" item."""
        raise NotImplementedError('"_compute_loss" method should '
                                  'be implemented in a child class.')

    def _compute_error(self, batch):
        """Computes error and adds it to "batch" as "validation_error" item."""
        raise NotImplementedError('"_compute_error" method should '
                                  'be implemented in a child class.')

    def _visualize_validation_errors(self, batch, batch_index):
        """Saves visualization of validation errors."""
        raise NotImplementedError(
            '"_visualize_validation_errors" method should '
            'be implemented in a child class.')

    def _train_for_epoch(self):
        """Returns training set losses."""
        self._network.train()
        training_losses = []
        number_of_batches = len(self._training_set_loader)
        for batch_index, batch in enumerate(self._training_set_loader):
            if _is_logging_required(batch_index, number_of_batches):
                logging.info('epoch {0:02d} ({1:02d}) : '
                             'training: {2:05d} ({3:05d})'.format(
                                 self._current_epoch + 1, self._end_epoch,
                                 batch_index + 1, number_of_batches))
            self._optimizer.zero_grad()
            if _is_on_cuda(self._network):
                batch = _move_tensors_to_cuda(batch)
            self._run_network(batch)
            self._compute_loss(batch)
            loss = batch['loss']
            loss.backward()
            self._optimizer.step()
            training_losses.append(loss.detach())
            del batch, loss
            th.cuda.empty_cache()
        return training_losses

    def _validate(self):
        """Returns validation set errors."""
        self._network.eval()
        validation_errors = []
        number_of_examples = len(self._validation_set_loader)
        for example_index, example in enumerate(self._validation_set_loader):
            if _is_logging_required(example_index, number_of_examples):
                logging.info('epoch: {0:02d} ({1:02d}) : '
                             'validation: {2:05d} ({3:05d})'.format(
                                 self._current_epoch + 1, self._end_epoch,
                                 example_index + 1, number_of_examples))
            if _is_on_cuda(self._network):
                example = _move_tensors_to_cuda(example)
            with th.no_grad():
                self._run_network(example)
            self._compute_error(example)
            validation_error = example['validation_error']
            validation_errors.append(validation_error)
            self._visualize_validation_errors(example, example_index)
            del example
            th.cuda.empty_cache()
        return validation_errors


class PdsTrainer(_Trainer):
    def _define_files(self):
        super(PdsTrainer, self)._define_files()
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
        # Here network output contains matching cost.
        batch['loss'] = self._criterion(batch['network_output'],
                                        batch['disparity_image'])

    def _compute_error(self, example):
        # Here network output contains disparity image.
        (example['pixelwise_validation_error'],
         example['validation_error']) = errors.compute_n_pixels_error(
             example['network_output'], example['disparity_image'])

    def _visualize_validation_errors(self, example, example_index):
        """Save visualization for 3 validation examples."""
        if example_index <= 3:
            # Dataset loader adds additional singletone dimension at the
            # beggining of tensors.
            ground_truth_disparity_image = example['disparity_image'][0].cpu()
            left_image = example['left_image'][0].cpu().byte()
            estimated_disparity_image = example['network_output'][0].cpu()
            pixelswise_3_pixels_error = example['pixelwise_validation_error'][
                0].cpu().byte()
            visualization.save_image(
                filename=self._left_image_template.format(example_index + 1),
                image=left_image)
            # Ensures same scale of the ground truth and estimated disparity.
            minimum_disparity = ground_truth_disparity_image.min()
            maximum_disparity = ground_truth_disparity_image.max()
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
                left_image, pixelswise_3_pixels_error)
            visualization.save_image(
                filename=self._3_pixels_error_image_template.format(
                    example_index + 1, self._current_epoch + 1),
                image=image_overlayed_with_errors)
