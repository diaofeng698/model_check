import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import time
import os
import random

import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


map_list = ['safe_driving', 'drinking', 'eating',  'smoking', 'phonecall',
            'phonetext']

weights_path = 'DAD_weights_6_11'

model_load = tf.keras.models.load_model(weights_path)

cap = cv2.VideoCapture('dms_res_sglee.avi')

fps = cap.get(cv2.CAP_PROP_FPS)
print(f'frame per second is {fps}')

num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(f'num frame is {num_frames}')

video_length = round(num_frames / fps)
print(f'video time length is {video_length}s')

frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
print(f'frame height is {frame_height}. width is {frame_width}')


while cap.isOpened():
    ret, frame = cap.read()
    if ret == False:
        print('read video failed or complete, break')
        break

    frame_now = cap.get(cv2.CAP_PROP_POS_FRAMES)

    time_start = time.time()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.resize(frame, (300, 300))
    gray_img = np.repeat(gray_img[..., np.newaxis], 3, -1)

    img_infer = model_load.predict(gray_img[np.newaxis, ...])
    img_pred = np.argmax(img_infer)

    conf = img_infer[0][img_pred]

    time_end = time.time()
    time_cost = round(time_end - time_start, 3)

    state_now = img_pred
    output_text = f'current frame is {frame_now}, infer result is {map_list[img_pred]}, infer time is {time_cost}'
    print(output_text)

    img_text = 'Result: ' + map_list[img_pred] + '   Conf: ' + str(conf)
    print(img_text)

    cv2.putText(frame, img_text, (50, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    # cv2.namedWindow("video")
    # cv2.imshow("video", frame)
    # cv2.waitKey(10)

cap.release()
