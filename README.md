# Context-Aware Grasp Detection for Target Objects in Cluttered Scenes Using Deep Hough Voting

![teaser](https://github.com/hoangcuongbk80/votegrasps/tree/master/doc/teaser.png)

## Introduction
This repository is code release for our VoteGrasp paper.

In this repository, we provide VoteGrasp model implementation (with Pytorch).

## Installation

Install [Pytorch](https://pytorch.org/get-started/locally/) and [Tensorflow](https://github.com/tensorflow/tensorflow) (for TensorBoard). It is required that you have access to GPUs. The code is tested with Ubuntu 16.04, Pytorch v1.1, TensorFlow v1.14, CUDA 10.0 and cuDNN v7.4.

Compile the CUDA layers for [PointNet++](http://arxiv.org/abs/1706.02413), which we used in the backbone network:

    cd pointnet2
    python setup.py install

To see if the compilation is successful, try to run `python models/votenet.py` to see if a forward pass works.

Install the following Python dependencies (with `pip install`):

    matplotlib
    opencv-python
    plyfile
    'trimesh>=2.35.39,<2.35.40'
    'networkx>=2.2,<2.3'

## Training

### Data preparation

Follow the [README](https://github.com/hoangcuongbk80/votegrasps/tree/master/ycbgrasp/README.md) under the `ycbgrasp` folder.

### Train

To train a new VoteGrasp model on the ycbgrasp data (synthetic):

    CUDA_VISIBLE_DEVICES=0 python train.py --dataset ycbgrasp --log_dir log_ycbgrasp

You can use `CUDA_VISIBLE_DEVICES=0,1,2` to specify which GPU(s) to use. Without specifying CUDA devices, the training will use all the available GPUs and train with data parallel (Note that due to I/O load, training speedup is not linear to the nubmer of GPUs used). Run `python train.py -h` to see more training options.
While training you can check the `log_ycbgrasp/log_train.txt` file on its progress.

## Run predict

    python predict.py

### Create and Train on your own data

If you have your own objects with 3D meshes, you can create a new dataset for your objects using our tools:
    [blender-scipts](https://github.com/votegrasp/blender-scripts)

## Acknowledgements
Will be available when our paper published.

## License
Will be available when our paper published.