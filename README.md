# Gazenet: head pose estimation without keypoints in MXNET/GLUON

We adopted the no-keypoints approach and reimplemented Ruiz et al., 2018 algorithm in
MXNET/GLUON (hereafter gazenet). 

### Why GLUON? 

Compared to other deep learning platforms, MXNET (with
GLUON API) not only provides the same simplicity and flexibility as Pytorch, but also allows
data scientists to hybridize the deep learning networks to leverage performance optimizations of
the symbolic graph. Moreover, MXNET/GLUON does not need to specify the input size of
networks, instead, it directly specifies the activation functions in the fully connected and the
convolutional layers, and it can create a namescope to attach a unique name to each layer.
Finally, its scalability and stability attract many retail companies to select MXNET/GLUON
platform for their product deployment.

### How gazenet work?

This gazenet algorithm takes in 3-channel (RGB) images and outputs three unit vectors of a
person's gazing direction, that is, yaw, roll, and pitch. The bounding box of
that person’s face is provided by a face detector we modified and trained based on the paper of
Najibi et al., 2017. Given the bounding box of a face, gazenet can detect that person’s gazing
directions even when that person is looking sideways and when the video or images are in
relatively low resolution, where the face landmarks are hard to detect thus the keypoint
approach tends to be fragile.

The original Pytorch algorithm can be found here: https://github.com/natanielruiz/deep-head-pose.

### References:

```
@InProceedings{Ruiz_2018_CVPR_Workshops,
author = {Ruiz, Nataniel and Chong, Eunji and Rehg, James M.},
title = {Fine-Grained Head Pose Estimation Without Keypoints},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2018}
}
```

```
@InProceedings{Shao_2019_FG,
author = {M Shao, Z Sun, M Ozay, T Okatani},
title = {Improving Head Pose Estimation with a Combined Loss and Bounding Box Margin Adjustment},
booktitle = {2019 14th IEEE International Conference on Automatic Face & Gesture Recognition (FG 2019)},
month = {April},
year = {2019}
}
```

```
@inproceedings{najibi2017ssh,
title={{SSH}: Single Stage Headless Face Detector},
author={Najibi, Mahyar and Samangouei, Pouya and Chellappa, Rama and Davis, Larry},
booktitle={The IEEE International Conference on Computer Vision (ICCV)},
year={2017}
}
```

