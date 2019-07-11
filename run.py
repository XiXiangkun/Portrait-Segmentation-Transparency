import datetime
start = datetime.datetime.now()

import os
from io import BytesIO
import tarfile
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2

# 解析模型，输入图片，返回resized图片和标签矩阵
class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    graph_def = None
    # Extract frozen graph from tar archive.
    tar_file = tarfile.open(tarball_path)
    for tar_info in tar_file.getmembers():
      if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
        file_handle = tar_file.extractfile(tar_info)
        graph_def = tf.GraphDef.FromString(file_handle.read())
        break

    tar_file.close()

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    return resized_image, seg_map

# 下两个函数根据标签设置颜色分割图
def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap

def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]

# 以第一个像素为准，相同色改为透明
def transparent_back(img):
    img = img.convert('RGBA')
    L, H = img.size
    color_0 = np.array([0, 0, 0, 255])
    for h in range(H):
        for l in range(L):
            dot = (l,h)
            color_1 = img.getpixel(dot)
            if (color_1 == color_0).all():
                color_1 = color_1[:-1] + (0,)
                img.putpixel(dot,color_1)
    return img


# plt显示多个图，包括原图和取出人像后的图，非人背景设置为黑色
def vis_segmentation(image, seg_map, url, back_url):
  """Visualizes input image, segmentation map and overlay view."""
  plt.figure(figsize=(15, 5))
  grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 6])

  plt.subplot(grid_spec[0])
  plt.imshow(image)
  plt.axis('off')
  plt.title('input image')

  plt.subplot(grid_spec[1])
  seg_image = label_to_color_image(seg_map).astype(np.uint8)
  seg_image = np.where(seg_image!=192,0,192)
  height,width,whatever = seg_image.shape
  # 载入原图
  image2 = cv2.imread(url)
  # cv2使用bgr方式读图，plt使用rgb
  b,g,r = cv2.split(image2)
  image3 = cv2.merge([r,g,b])
  image3 = cv2.resize(image3,(width,height))
  # 载入背景图
  ima = cv2.imread(back_url)
  b2, g2, r2 = cv2.split(ima)
  ima2 = cv2.merge([r2, g2, b2])
  re_image = cv2.resize(ima2, (width, height))

  for ia in range(0,height):
    for ib in range(0,width):
      if (seg_image[ia][ib]==np.array([0,0,0])).all():
        image3[ia][ib]=[0,0,0]
      else:
        re_image[ia][ib]=image3[ia][ib]

  plt.imshow(image3)
  # cv2.imwrite希望输入的图片是BGR格式
  image3 = cv2.cvtColor(image3, cv2.COLOR_RGB2BGR)
  cv2.imwrite("./temp_images/pre_image.png",image3)
  plt.axis('off')
  plt.title('image segmentation')

  plt.subplot(grid_spec[2])
  plt.imshow(re_image)
  #plt.imshow(seg_image, alpha=0)
  plt.axis('off')
  plt.title('background changed')
  re_image = cv2.cvtColor(re_image, cv2.COLOR_RGB2BGR)
  cv2.imwrite('./temp_images/re_image.png',re_image)


  image4 = Image.open("./temp_images/pre_image.png")
  image_transparent = transparent_back(image4)
  plt.subplot(grid_spec[3])
  plt.imshow(image_transparent)
  #plt.imshow(seg_image, alpha=0)
  plt.axis('off')
  plt.title('transparent image')
  image_transparent.save('./temp_images/image_transparent.png')

  plt.show()

# 标签属性
LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])

def run_visualization(url, back_url):
  try:
    original_im = Image.open(url)
  except IOError:
    print('Cannot retrieve image. Please check url: ' + url)
    return

  print('running deeplab on image %s...' % url)
  resized_im, seg_map = MODEL.run(original_im)
  vis_segmentation(resized_im, seg_map, url, back_url)


###############需要修改：模型地址###############################################
download_path = 'models/download/mobilenetv2/deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz'
MODEL = DeepLabModel(download_path)
print('model loaded successfully!')
load_time = datetime.datetime.now()
print('load时间：',load_time-start)
###############需要修改：图片所在地址###########################################
image_url = './test_images/test.jpg'
back_url = './test_images/back.jpg'
run_visualization(image_url, back_url)

end = datetime.datetime.now()
print('运行时间：',end-start)