# Gazenet: head pose estimation without keypoints in GLUON

In retail, sales data is commonly used to identify hot products in stores for marketing. For instance, products that are sold well in other stores are usually marketed to new stores. With the increasingly successful use cases of machine deep learning, especially Convolutional Neural Networks (CNN) in computer vision, however, companies have been combining new insights extracted from images with sales data for marketing strategies. In this blog, we are going to discuss a new set of information from a head pose estimation with Euler angles in MXNet/Gluon.

Head pose estimation from an image is currently derived from two main methods: with and without facial keypoints, which include eyes, ears, and nose. The accuracy of the keypoints approach depends upon the correct representation of a 3D generic body model, which is usually difficult to achieve. The no-keypoints approach, however, works around the depth complexity of the keypoints approach and directly learns from the 2D images with multi-loss. For instance, Ruiz et al., 2018 and Shao et al., 2019 developed no-keypoints models in Pytorch and Tensorflow, respectively, and their models outperform the traditional face landmark algorithms on several widely used data sets.

We adopted the no-keypoints approach and reimplemented Ruiz et al., 2018 algorithm in MXNet/Gluon, hereafter gazenet. Compared to other deep learning platforms, MXNet (with Gluon API) not only provides the same simplicity and flexibility as Pytorch but also allows data scientists to hybridize the deep learning networks to leverage performance optimizations of the symbolic graph. Moreover, MXNet/Gluon does not need to specify the input size of networks, instead, it directly specifies the activation functions in the fully connected and the convolutional layers, and it can create a name scope to attach a unique name to each layer. Finally, its scalability and stability attract many retail companies to select MXNet/Gluon platform for their product deployment.

This gazenet algorithm takes in 3-channel (RGB) images and outputs three unit vectors of a person’s gazing direction, that is, yaw, roll, and pitch, as illustrated in Fig. 1. The bounding box of that person’s face is provided by a face detector we modified and trained based on the paper of Najibi et al., 2017. Given the bounding box of a face, gazenet can detect that person’s gazing directions even when that person is looking sideways and when the video or images are in relatively low resolution, where the face landmarks are hard to detect thus the keypoints approach tends to be fragile.

Similar to Ruiz et al., 2018 and Shao et al., 2019, gazenet employs a pre-trained ResNet50 (He et al., 2015) architecture followed by a fully connected layer, and a softmax function is then used to derive the class scores. Multi-loss functions are used to classify and regress each angle to degrees. Its architecture is illustrated below. Gazenet achieves comparable performance as Ruiz et al., 2018 on the public data set of AFLW2000 with approximately 6.5 degrees average errors for yaw, roll, pitch, and mean squared error. Its open-sourced MXNet/Gluon implementation is here: https://github.com/Cjiangbpcs/gazenet_mxJiang/blob/master/README.md. A description of a video analytics product that adopted gazenet can be found here: https://bpcs.com/reflect.

The estimated Euler angles can be further aggregated to provide insights about hot product candidates at stores for marketing experiments. This new data, together with traditional sales data, can provide retail stores valuable information to design experiments and drive measurable business impacts.

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

@article{He2015,
	author = {Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
	title = {Deep Residual Learning for Image Recognition},
	journal = {arXiv preprint arXiv:1512.03385},
	year = {2015}
}
```

