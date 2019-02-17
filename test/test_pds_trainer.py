# Copyrights. All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.

import os
import pkg_resources
import tempfile

from torch import optim
from torch.optim import lr_scheduler
from torch.utils import data

from practical_deep_stereo import flyingthings3d_dataset
from practical_deep_stereo import loss
from practical_deep_stereo import pds_network
from practical_deep_stereo import pds_trainer
from practical_deep_stereo import transformers


FOLDER_WITH_FRAGMENT_OF_FLYINGTHINGS3D_DATASET = \
 pkg_resources.resource_filename(__name__, "data/flyingthings3d")


def _initialize_parameters():
    test_set = flyingthings3d_dataset.FlyingThings3D.benchmark_dataset(
        FOLDER_WITH_FRAGMENT_OF_FLYINGTHINGS3D_DATASET, True)
    test_set.subsample(1)
    training_set, validation_set = \
        flyingthings3d_dataset.FlyingThings3D.training_split(
            FOLDER_WITH_FRAGMENT_OF_FLYINGTHINGS3D_DATASET,
                number_of_validation_examples=1)
    transformers_list = [transformers.CentralCrop(height=64, width=64,
        get_items_to_crop=lambda x: [x['left']['image'],
        x['right']['image'], x['left']['disparity_image']])]
    training_set.append_transformers(transformers_list)
    validation_set.append_transformers(transformers_list)
    test_set.append_transformers(transformers_list)
    training_set_loader = data.DataLoader(
        training_set,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True)
    validation_set_loader = data.DataLoader(
        validation_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)
    test_set_loader = data.DataLoader(
        test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    network = pds_network.PdsNetwork()
    network.set_maximum_disparity(63)
    optimizer = optim.RMSprop(network.parameters(), lr=1e-3)
    return {
        'network':
        network,
        'optimizer':
        optimizer,
        'criterion':
        loss.SubpixelCrossEntropy(),
        'learning_rate_scheduler':
        lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5),
        'training_set_loader':
        training_set_loader,
        'test_set_loader':
        test_set_loader,
        'validation_set_loader':
        validation_set_loader,
        'end_epoch':
        2,
        'experiment_folder':
        tempfile.mkdtemp()
    }


def test_pds_trainer():
    trainer = pds_trainer.PdsTrainer(_initialize_parameters())
    trainer.train()
    assert len(trainer._training_losses) == 2
    assert trainer._current_epoch == 2
    checkpoint_file = os.path.join(trainer._experiment_folder,
                                   '002_checkpoint.bin')
    trainer = pds_trainer.PdsTrainer(_initialize_parameters())
    trainer.load_checkpoint(checkpoint_file)
    trainer._current_epoch == 2
    trainer._end_epoch = 3
    trainer.train()
    assert len(trainer._training_losses) == 3
    assert trainer._current_epoch == 3
    assert trainer._training_losses[0] > trainer._training_losses[2]
    trainer.test()
