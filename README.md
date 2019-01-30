# Practical Deep Stereo (Work in progress.. Please wait for pre-trained models.) 
This repository contains refactored code for ["Practical Deep Stereo (PDS): Toward applications-friendly deep stereo matching" by Stepan Tulyakov, Anton Ivanov and Francois Fleuret](https://papers.nips.cc/paper/7828-practical-deep-stereo-pds-toward-applications-friendly-deep-stereo-matching), that appeared on NeurIPS2018 as a poster.

## Preparing Datasets
To set up FlyingThings3D dataset, download PNG RGB cleanpass images and disparities from [website of Patter Recognition and Image Processing group of University of Freiburg](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html). Unpack the archive with images into `PracticalDeepStereo_NIPS2018/datasets/flyingthings3d/frames_cleanpass` and archive with disparities into `PracticalDeepStereo_NIPS2018/datasets/flyingthings3d/disparity`.      

## Training and benchmarking on FlyingThings3D
To run training on Flyingthings3D run
```
./train_on_flyingthings3d.py \
--experiment_folder experiments/flyingthings3d \
--dataset_folder datasets/flyingthings3d \
```
During the first run, the dataset object calculates and saves disparity statistic for every example in the dataset. Therefore, it might take a while before actual training starts. Overall, the training on full-size image is very slow since it does not use batch processing. The training can be started from a checkpoint by setting the `--checkpoint_file` flag.

To benchmark on Flyingthings3D run
```
./benchmark_on_flyingthings3d.py \
--experiment_folder experiments/flyingthings3d \
--dataset_folder datasets/flyingthings3d \
--checkpoint_file experiments/flyingthings3d/008_checkpoint.bin \
--is_psm_protocol
```
The evalutaion protocol can be selected by setting / unsetting the `--is_psm_protocol` flag.

## Troubleshooting
If one of the training scripts does not work please run all unit tests by executing `./run_unit_tests.sh`. This will help you to localize and fix bugs on your own.  
