import warnings
warnings.filterwarnings('ignore')  # 不打印 warning

import tensorflow as tf


# 设置GPU按需增长
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

import numpy as np

X = np.arange(20).reshape([10, 2])
y = np.arange(10).reshape([10, 1])

dataset = tf.data.Dataset.from_tensor_slices((X, y))
print(dataset)

iterator = dataset.make_one_shot_iterator()
print(iterator)

import sys

sess.run(tf.global_variables_initializer())
count = 0
try:
    while True:
        x_bath, y_batch = sess.run(iterator.get_next())
        print('count= {}: x= {} y= {}'.format(count, x_bath,y_batch))
        count += 1

except Exception as e:
    print('\n', e)
    print('final count = {}'.format(count))