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

frame = cv2.imread("chuanchuan_phone_interact_1MU9HI.jpg")

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

img_text = 'Result: ' + map_list[img_pred] + '   Conf: ' + str(conf)
print(img_text)

cv2.putText(frame, img_text, (50, 240),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
# cv2.namedWindow("video")
# cv2.imshow("video", frame)
# cv2.waitKey(10)
