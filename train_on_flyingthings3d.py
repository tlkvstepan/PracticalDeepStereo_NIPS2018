#!/usr/bin/env python
# Copyrights. All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.
"""Script performs trainig from scratch on flyingthings3D.

Training is performed with maximum disparity of 255 on
960 x 540 full-size images without any augmentation.

For optimization RMSprop method with standard setting is used.
Training is performed for 160k iterations in totla.
During first 120k iterations learning rate is set to 0.01
and than it is halfed every 20k iterations.

500 examples from the canonical training set are allocated
for validation and the rest of the examples are allocated for
training dataset.

All images with disparities outside of [0, 255] disparity
range are excluded from training. Images with rendering
artifacts are also excluded.

Optionally, the user can pass to the script:
"dataset_folder" with flyinghtings3d dataset;
"experiment_folder" where experiment results are be saved;
"checkpoint_file" with checkpoint that will be loaded
                  to restart training.

Example call:

./train_on_flyingthings3d.py \
--experiment_folder experiments/flyingthings3d \
--dataset_folder datasets/flyingthings3d \
--checkpoint_file experiments/flyingthings3d/001_checkpoint.bin
"""

import os
import click

from torch import optim
from torch.optim import lr_scheduler
from torch.utils import data

from practical_deep_stereo import flyingthings3d_dataset
from practical_deep_stereo import loss
from practical_deep_stereo import pds_network
from practical_deep_stereo import trainer


def _initialize_parameters(dataset_folder, experiment_folder):
    training_set, validation_set = \
        flyingthings3d_dataset.FlyingThings3D.training_split(
                dataset_folder)
    training_set_loader = data.DataLoader(
        training_set,
        batch_size=1,
        shuffle=True,
        num_workers=3,
        pin_memory=True)
    validation_set_loader = data.DataLoader(
        validation_set,
        batch_size=1,
        shuffle=False,
        num_workers=3,
        pin_memory=True)
    network = pds_network.PdsNetwork().cuda()
    optimizer = optim.RMSprop(network.parameters(), lr=1e-2)
    # Learning rate is 1e-2 for first 120k iterations, and than
    # is halfed every 20k iterations.
    learning_rate_scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=[6, 7, 8, 9, 10], gamma=0.5)
    criterion = loss.SubpixelCrossEntropy().cuda()
    return {
        'network': network,
        'optimizer': optimizer,
        'criterion': criterion,
        'learning_rate_scheduler': learning_rate_scheduler,
        'training_set_loader': training_set_loader,
        'validation_set_loader': validation_set_loader,
        'end_epoch': 10,
        'experiment_folder': experiment_folder
    }


@click.command()
@click.option(
    '--dataset_folder',
    default='datasets/flyingthings3d',
    type=click.Path(exists=True))
@click.option(
    '--experiment_folder',
    default='experiments/flyingthings3d',
    type=click.Path(exists=False))
@click.option('--checkpoint_file', default=None, type=click.Path(exists=True))
def train_on_flyingthings3d(dataset_folder, experiment_folder,
                            checkpoint_file):
    dataset_folder = os.path.abspath(dataset_folder)
    experiment_folder = os.path.abspath(experiment_folder)
    if not os.path.isdir(experiment_folder):
        os.mkdir(experiment_folder)
    pds_trainer = trainer.PdsTrainer(
        _initialize_parameters(dataset_folder, experiment_folder))
    if checkpoint_file is not None:
        checkpoint_file = os.path.abspath(checkpoint_file)
        pds_trainer.load_checkpoint(checkpoint_file)
    pds_trainer.train()


if __name__ == '__main__':
    train_on_flyingthings3d()
