# Practical Deep Stereo 
This repository contains refactored code for ["Practical Deep Stereo (PDS): Toward applications-friendly deep stereo matching" by Stepan Tulyakov, Anton Ivanov and Francois Fleuret](https://papers.nips.cc/paper/7828-practical-deep-stereo-pds-toward-applications-friendly-deep-stereo-matching), that appeared on NeurIPS2018 as a poster.


## Preparing Datasets
To set up FlyingThings3D dataset, download PNG RGB cleanpass images and disparities from [website of Patter Recognition and Image Processing group of University of Freiburg](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html). Unpack the archive with images into `PracticalDeepStereo_NIPS2018/datasets/flyingthings3d/frames_cleanpass` and archive with disparities into `PracticalDeepStereo_NIPS2018/datasets/flyingthings3d/disparity`.      

## Training on FlyingThings3D
During the first run, the dataset object calculates and saves disparity statistic for every example in the dataset. Therefore, it might take a while before actuall training starts.

## Troubleshooting
If one of the training scripts does not work please run all unit tests by executing `./run_unit_tests.sh`. This will help you to localize and fix bugs on your own.  

(Work in progress)
