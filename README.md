### deeplabv3（mobilenetv2+xception）加载model，分割测试图片并更换背景，设置背景透明化：
- #### 使用：
  - model文件过大，超过400M无法上传，需要自己创建目录文件并下载，下载地址：`https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md`，修改目录结构，目录结构放在文件末尾。
  - 目录结构：
   
  ```
  D:.  
  │  README.md  
  │  run.py  
  │  
  ├─models  
  │  └─download  
  │      ├─mobilenetv2  
  │      │      deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz
  │      │      deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz
  │      │
  │      └─xception
  │              deeplabv3_pascal_trainval_2018_01_04.tar.gz
  │              deeplabv3_pascal_train_aug_2018_01_04.tar.gz
  │
  ├─temp_images
  │      image_transparent.png
  │      pre_image.png
  │      re_image.png
  │
  └─test_images
          back.jpg
          image1.jpg
          test.jpg
          test_people.jpg
  ```
  - 修改run.py代码末尾的参数，选择model和图片，运行
  - 依赖：
    - numpy
    - Image
    - cv2
    - tensorflow
    - matplotlib
    - tarfile

- #### 运行结果：
  - 四次对比：
    - mobilenetv2_aug:  
<img src="https://github.com/XiXiangkun/images/blob/master/mobilenetv2_aug.png?raw=true" width="500" hegiht="250" align=center />
    - mobilenetv2_val:  
<img src="https://github.com/XiXiangkun/images/blob/master/mobilenetv2_val.png?raw=true" width="500" hegiht="250" align=center />
    - xception_aug:  
<img src="https://github.com/XiXiangkun/images/blob/master/xception_aug.png?raw=true" width="500" hegiht="250" align=center />
    - xception_val:  
<img src="https://github.com/XiXiangkun/images/blob/master/xception_val.png?raw=true" width="500" hegiht="250" align=center />
  - 最终效果：  
<img src="https://github.com/XiXiangkun/images/blob/master/seg_result.png?raw=true" width="700" hegiht="350" align=center />


- #### model的运行时间对比:
  - mobilenetv2:  

|类型           | aug           | val  |
| :-------------: |:-------------:| :-----:|
| load时间      | 0:00:02.202638 | 0:00:05.498740 |
| 运行时间      | 0:00:05.233193      |   0:00:08.560493 |

- 
  - xception:

|类型           | aug           | val  |
| :-------------: |:-------------:| :-----:|
| load时间      | 0:00:09.372786 | 0:00:09.372795 |
| 运行时间      | 0:00:26.899916      |   0:00:26.540638 |

- #### model的其他方面对比：
  - 初次运行：  
    - load时间： 0:00:34.842048
    - 运行时间： 0:00:41.793574
  - Xception和MobileNet都是采用depthwise separable convolution
    - 但是二者的目的是不同的，Xception使用depthwise separable convolution的同时增加了网络的参数量来比对效果，主要考察这种结构的有效性
    - MobileNet则是用depthwise separable convolution来进行压缩和提速的，参数量明显减少，目的也不在性能提升上面而是速度上面。
  - 准确率方面，从上往下的顺序mIOU依次增大，分割效果更好。官网对比介绍：`https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md`
  - 使用的4个model都是的在PASCAL VOC 2012数据集上训练出的结果，官方还有基于Cityscapes和ADE20K数据集训练出来的model和基于ImageNet的预训练模型，但是训练结果并不好。
  
- #### deeplab models训练部分（xception）：
  - 下载数据集转成TFRecord
  - 设置参数：
    - `training_number_of_steps`训练30000次
    - `atrous_rates`设置[6,12,18]；`atrous_convolution`可以理解为带洞的卷积，如果 rate 参数是1，则表现为普通的 2-d 卷积。如果 rate 参数大于1，则会表现则带有洞的卷积。这里并没有深入探索
    - `output_stride`输出步长为16
    - `decoder_output_stride`解码输出步长设为4
    - `train_crop_size`用于训练和测试的图片都是513*513大小
    - `train_batch_size`训练的批量大小为1（要实现tensorflow提供models的准确度，需要用大批量>12和`fine_tune_batch_norm`微调批次定额来训练）
    - `PATH_TO_INITIAL_CHECKPOINT`设置使用MS-COCO训练集预训练的ImageNet结果
- #### deeplab的结构部分内容：
  - 图像分割可以分为两类：语义分割（Semantic Segmentation）和实例分割（Instance Segmentation）
  - 与检测模型类似，语义分割模型也是建立是分类模型基础上的，即利用CNN网络来提取特征进行分类。对于CNN分类模型，一般情况下会存在stride>1的卷积层和池化层来降采样，此时特征图维度降低，但是特征更高级，语义更丰富。这对于简单的分类没有问题，因为最终只预测一个全局概率，对于分割模型就无法接受，因为我们需要给出图像不同位置的分类概率，特征图过小时会损失很多信息。其实对于检测模型同样存在这个问题，但是由于检测比分割更粗糙，所以分割对于这个问题更严重。但是下采样层又是不可缺少的，首先stride>1的下采样层对于提升感受野非常重要，这样高层特征语义更丰富，而且对于分割来说较大的感受野也至关重要；另外的一个现实问题，没有下采样层，特征图一直保持原始大小，计算量是非常大的。相比之下，对于前面的特征图，其保持了较多的空间位置信息，但是语义会差一些，但是这些空间信息对于精确分割也是至关重要的。这是语义分割所面临的一个困境或者矛盾，也是大部分研究要一直解决的。
  - 对于这个问题，主要存在两种不同的解决方案：
    - 一种是原始的FCN（Fully Convolutional Networks for Semantic Segmentation），图片送进网络后会得到小32x的特征图，虽然语义丰富但是空间信息损失严重导致分割不准确，这称为FCN-32s，另外paper还设计了FCN-8s，大致是结合不同level的特征逐步得到相对精细的特征，效果会好很多。为了得到高分辨率的特征
    - 一种更直观的解决方案是EncoderDecoder结构，其中Encoder就是下采样模块，负责特征提取，而Decoder是上采样模块（通过插值，转置卷积等方式），负责恢复特征图大小，一般两个模块是对称的，经典的网络如U-Net（U-Net: Convolutional Networks for Biomedical Image Segmentation。而要直接将高层特征图恢复到原始大小是相对困难的，所以Decoder是一个渐进的过程，而且要引入横向连接（lateral connection），即引入低级特征增加空间信息特征分割准确度，横向连接可以通过concat或者sum操作来实现。
    - 另外一种结构是DilatedFCN，主要是通过空洞卷积（Atrous Convolution）来减少下采样率但是又可以保证感受野，如图中的下采样率只有8x，那么最终的特征图语义不仅语义丰富而且相对精细，可以直接通过插值恢复原始分辨率。天下没有免费的午餐，保持分辨率意味着较大的运算量，这是该架构的弊端。DeepLabv3+就是属于典型的DilatedFCN，它是Google提出的DeepLab系列的第4弹。
- #### 参考：
  `https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/pascal.md`  
  `https://zhuanlan.zhihu.com/p/62261970`

  
