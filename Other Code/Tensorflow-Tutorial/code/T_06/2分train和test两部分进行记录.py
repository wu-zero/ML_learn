import warnings
warnings.filterwarnings('ignore')  # 不打印 warning

import tensorflow as tf

# 设置GPU按需增长
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

import numpy as np
import os
import shutil

log_dir = './graph2/'
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)

os.makedirs(log_dir)
print('created log_dir path')

a = tf.placeholder(dtype=tf.float32, shape=[100, 1], name='a')

with tf.name_scope('add_example'):
    b = tf.Variable(tf.truncated_normal([100, 1], mean=-0.5, stddev=1.0), name='var_b')
    tf.summary.histogram('b_hist', b)
    increase_b = tf.assign(b, b + 0.2)
    c = tf.add(a, b)
    tf.summary.histogram('c_hist', c)
    c_mean = tf.reduce_mean(c)
    tf.summary.scalar('c_mean', c_mean)
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)  # 保存位置
test_writer = tf.summary.FileWriter(log_dir + '/test', sess.graph)


sess.run(tf.global_variables_initializer())
for step in range(500):
    if (step+1) % 10 == 0:
        _a = np.random.randn(100, 1)
        summary, _ = sess.run([merged, increase_b], feed_dict={a: _a})    # 每步改变一次 b 的值
        test_writer.add_summary(summary, step)
    else:
        _a = np.random.randn(100, 1) + step*0.2
        summary, _ = sess.run([merged, increase_b], feed_dict={a: _a})    # 每步改变一次 b 的值
        train_writer.add_summary(summary, step)
train_writer.close()
test_writer.close()
print('END!')