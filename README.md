### deeplabv3（mobilenetv2+xception）加载model，分割测试图片并更换背景，设置背景透明化：

- #### 使用：
  - 修改run.py代码末尾的参数，选择model和图片，运行
  - 依赖：
    - numpy
    - Image
    - cv2
    - tensorflow
    - matplotlib
    - tarfile
- #### model的运行时间对比:
  - mobilenetv2:  
  |类型|aug|val|
  | :---: | :----: | :-----: |
  |load时间|0:00:02.202638|0:00:05.498740|
  |运行时间|0:00:05.233193|0:00:08.560493|

  - xception:
  |类型|aug|val|
  | :---: | :----: | :-----: |
  |load时间|0:00:09.372786|0:00:09.372795|
  |运行时间|0:00:26.899916|0:00:26.540638|

- #### model的其他方面对比：
  - 初次运行：  
    - load时间： 0:00:34.842048
    - 运行时间： 0:00:41.793574
  - Xception和MobileNet都是采用depthwise separable convolution
    - 但是二者的目的是不同的，Xception使用depthwise separable convolution的同时增加了网络的参数量来比对效果，主要考察这种结构的有效性
    - MobileNet则是用depthwise separable convolution来进行压缩和提速的，参数量明显减少，目的也不在性能提升上面而是速度上面。
  - 准确率方面，从上往下的顺序mIOU依次增大，分割效果更好。官网对比介绍：`https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md`
  - 使用的4个model都是的在PASCAL VOC 2012数据集上训练出的结果，官方还有基于Cityscapes和ADE20K数据集训练出来的model和基于ImageNet的预训练模型，但是训练结果并不好。


- #### 运行结果：
  - 四次对比：
    - mobilenetv2_aug:  
<img src="https://github.com/XiXiangkun/images/blob/master/mobilenetv2_aug.png?raw=true" width="400" hegiht="150" align=center />
    - mobilenetv2_val:  
<img src="https://github.com/XiXiangkun/images/blob/master/mobilenetv2_val.png?raw=true" width="400" hegiht="150" align=center />
    - xception_aug:  
<img src="https://github.com/XiXiangkun/images/blob/master/xception_aug.png?raw=true" width="400" hegiht="150" align=center />
    - xception_val:  
<img src="https://github.com/XiXiangkun/images/blob/master/xception_val.png?raw=true" width="400" hegiht="150" align=center />
  - 最终效果：  
<img src="https://github.com/XiXiangkun/images/blob/master/seg_result.png?raw=true" width="400" hegiht="150" align=center />
