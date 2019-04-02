## Code for CVPR 2019 Paper "[Hybrid-Attention based Decoupled Metric Learning for Zero-Shot Image Retrieval](http://bhchen.cn)"

This code is developed based on [Caffe](https://github.com/BVLC/caffe/).

* [Updates](#updates)
* [Files](#files)
* [Prerequisites](#prerequisites)
* [Train_Model](#train_model)
* [Extract_DeepFeature](#extract_deepfeature)
* [Contact](#contact)
* [Citation](#citation)
* [LICENSE](#license)
* [README_Caffe](#readme_caffe)

### Updates
- Apr 02, 2019
  * The code and training prototxt for our [CVPR19](http://bhchen.cn) paper are released.
  * For simplication, we use Hinge_Loss-like adversarial constraint instead of the original adversary network (This will reduce the final performance a little but ease the training. We also provide the "gradients_reverse_layer" for the implementation of the orginal adversary network, you can try it by yourself.).
  * If you train our Network on **CUB-200**, the expected retrieval performance of DeML(I=3,J=3) will be ~**(R@1=64.7, R@2=75.1, R@4=83.2, R@8=89.4)** at ~10k iterations.

### Files
- Original Caffe library
- BinomialLoss
  * src/caffe/proto/caffe.proto
  * include/caffe/layers/Binomial_loss_layer.hpp
  * src/caffe/layers/Binomial_loss_layer.cpp
- ActLoss
  * src/caffe/proto/caffe.proto
  * include/caffe/layers/act_loss.hpp
  * src/caffe/layers/act_loss.cpp
- OAM
  * src/caffe/proto/caffe.proto
  * include/caffe/layers/proposal_crop_layer.hpp (For single convolution input)
  * src/caffe/layers/proposal_crop_layer.cpp (For single convolution input)
  * include/caffe/layers/proposal_crop_layer.hpp (For multi convolution input)
  * src/caffe/layers/proposal_crop_layer.cpp (For multi convolution input)
- gradients_reverse
  * src/caffe/proto/caffe.proto
  * include/caffe/layers/gradients_reverse_layer.hpp
  * src/caffe/layers/gradients_reverse_layer.cpp
  * src/caffe/layers/gradients_reverse_layer.cu
- CUB
  * examples/CUB/U512  (This is for the implementation of baseline method U512)
  * examples/CUB/DeML3-3_512  (This is for the implementation of our method DeML(I=3,J=3))
  * examples/CUB/pre-trained-model
### Prerequisites
* Caffe
* Matlab (for evaluation)
* 2GPUs, each > 11G
### Train_Model
1. The Installation is completely the same as [Caffe](http://caffe.berkeleyvision.org/). Please follow the [installation instructions](http://caffe.berkeleyvision.org/installation.html). Make sure you have correctly installed before using our code. 
2. Download the training images [CUB]() and move it to $(your_path). The images are preprossed the same as [Lifted Loss](https://github.com/rksltnl/Deep-Metric-Learning-CVPR16/), i.e. with zero paddings.
3. Download the training_list (~400M), and move it ~/DeML/examples/CUB/ . (Or you can create your own list by randomly selecting 65 classes with 2 samples each class.)
4. Download googlenetV1 model (used for U512) and 3nets model (used for DeML3-3_512) to folder ~/DeML/examples/CUB/pre-trained-model/
5. Modify the images path by changing "root_folder" in all *.prototxt into $(your_path).
6. Then you can train our baseline method U512 and the proposed DeML3-3_512 by running
```
        cd ~/DeML/examples/CUB/U512
        ./finetune_U512.sh
```     
and
```
        cd ~/DeML/examples/CUB/DeML3-3_512
        ./finetune_DeML3-3_512.sh
```     
### Extract_DeepFeature
1. Compile matcaffe by make matcaffe
2. Specify the correspinding paths in face_example/extractDeepFeature.m

        addpath('path_to_matCaffe/matlab');
        model = 'path_to_deploy/face_deploy.prototxt';
        weights = 'path_to_model/face_model.caffemodel';
        image = imread('path_to_image/Jennifer_Aniston_0016.jpg');

3. Run extractDeepFeature.m in Matlab

### Contact 
- [Binghui Chen](http://bhchen.cn)

### Citation
You are encouraged to cite the following papers if this work helps your research. 

    @inproceedings{chen2017hybrid,
      title={Hybrid-Attention based Decoupled metric Learning for Zero-Shot Image Retrieval},
      author={Chen, Binghui and Deng, Weihong},
      booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2019},
    }
    @InProceedings{chen2019energy,
    author = {Chen, Binghui and Deng, Weihong},
    title = {Energy Confused Adversarial Metric Learning for Zero-Shot Image Retrieval and Clustering},
    booktitle = {AAAI Conference on Artificial Intelligence},
    year = {2019}
    }
    @inproceedings{songCVPR16,
    Author = {Hyun Oh Song and Yu Xiang and Stefanie Jegelka and Silvio Savarese},
    Title = {Deep Metric Learning via Lifted Structured Feature Embedding},
    Booktitle = {Computer Vision and Pattern Recognition (CVPR)},
    Year = {2016}
    }
### License
Copyright (c) Binghui Chen

All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

***

### README_Caffe
# Caffe

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BVLC reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
