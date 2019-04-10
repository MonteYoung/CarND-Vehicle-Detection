# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
from moviepy.editor import VideoFileClip
# from IPython.display import HTML

import keras # broken for keras >= 2.0, use 1.2.2
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Flatten, Dense, Activation, Reshape

from utils import load_weights, Box, yolo_net_out_to_car_boxes, draw_box
keras.backend.set_image_dim_ordering('th') # “th”格式意味着卷积核将具有形状（depth，input_depth，rows，cols）


def model():
    """
    创建网络模型
    :return: model
    """
    model = Sequential()
    model.add(Convolution2D(16, 3, 3,input_shape=(3,448,448),border_mode='same',subsample=(1,1)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(32,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
    model.add(Convolution2D(64,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
    model.add(Convolution2D(128,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
    model.add(Convolution2D(256,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
    model.add(Convolution2D(512,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
    model.add(Convolution2D(1024,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Convolution2D(1024,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Convolution2D(1024,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dense(4096))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(1470))
    model.summary()

    return model

# def test_a_pic():
#     """
#     弃用
#     :return:
#     """
#     imagePath = './test_images/test1.jpg'
#     image = plt.imread(imagePath)
#     image_crop = image[300:650,500:,:]
#     resized = cv2.resize(image_crop, (448,448))
#
#     batch = np.transpose(resized,(2,0,1))
#     batch = 2*(batch-0./255.- 0.) - 1   # normalize to -1 ~ 1
#     batch = np.expand_dims(batch, axis=0)
#     out = model.predict(batch)
#
#     boxes = yolo_net_out_to_car_boxes(out[0], threshold = 0.17)
#
#     f, (ax1,ax2) = plt.subplots(1,2,figsize=(16,6))
#     ax1.imshow(image)
#     ax2.imshow(draw_box(boxes, plt.imread(imagePath),[[500,1280],[300,650]]))
#     plt.show()

def test_batch_pic(path_pic="./test_images/*.jpg"):
    """
    测试多张图片
    :param path_pic: 图片所在路径
    :return:
    """
    images = [plt.imread(file) for file in glob.glob(path_pic)]

    # 图像的shape转化成 [c, h, w],
    # batch的shape为 [n, c, h, w],
    # 此处image[300:650,500:,:]是因为车辆都在图像右下方,
    # resize到 448*448 满足网络输入要求
    batch = np.array([np.transpose(cv2.resize(image[300:650,500:,:],(448,448)),(2,0,1))
                      for image in images])

    batch = 2*(batch/255.) - 1 # 归一化到 [-1.0, 1.0]
    out = model.predict(batch)

    for i in range(len(batch)):
        boxes = yolo_net_out_to_car_boxes(out[i], threshold = 0.17)
        plt.imshow(draw_box(boxes,images[i],[[500,1280],[300,650]]))
        plt.savefig("./output_images/test_output_pic_{}".format(i))
        plt.show()


def frame_func(image):
    crop = image[300:650,500:,:]
    resized = cv2.resize(crop,(448,448))
    batch = np.array([resized[:,:,0],resized[:,:,1],resized[:,:,2]])
    batch = 2*(batch/255.) - 1
    batch = np.expand_dims(batch, axis=0)
    out = model.predict(batch)
    boxes = yolo_net_out_to_car_boxes(out[0], threshold = 0.17)
    return draw_box(boxes,image,[[500,1280],[300,650]])

if __name__ == '__main__':
    model = model()
    # loading weights
    load_weights(model,'./yolo-tiny.weights')
    # test_a_pic()
    test_batch_pic()

    # project_video_output = './project_video_output.mp4'
    # clip1 = VideoFileClip("./project_video.mp4")
    #
    # lane_clip = clip1.fl_image(frame_func) #NOTE: this function expects color images!!
    # # %time
    # lane_clip.write_videofile(project_video_output, audio=False)