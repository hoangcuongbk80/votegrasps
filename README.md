# Context-Aware Grasp Detection for Target Objects in Cluttered Scenes Using Deep Hough Voting

![teaser](https://github.com/facebookresearch/votenet/blob/master/doc/teaser.jpg)

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

The demo uses a pre-trained model (on SUN RGB-D) to detect objects in a point cloud from an indoor room of a table and a few chairs (from SUN RGB-D val set). You can use 3D visualization software such as the [MeshLab](http://www.meshlab.net/) to open the dumped file under `demo_files/sunrgbd_results` to see the 3D detection output. Specifically, open `***_pc.ply` and `***_pred_confident_nms_bbox.ply` to see the input point cloud and predicted 3D bounding boxes.

You can also run the following command to use another pretrained model on a ScanNet:

    python demo.py --dataset scannet --num_point 40000

Detection results will be dumped to `demo_files/scannet_results`.

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