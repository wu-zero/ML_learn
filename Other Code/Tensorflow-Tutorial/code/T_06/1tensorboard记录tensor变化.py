import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


import os
import shutil
log_dir = './graph/'
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)

os.makedirs(log_dir)
print('created log_dir path')

with tf.name_scope('add_example'):
    a = tf.Variable(tf.truncated_normal([100, 1], mean=0.5, stddev=0.5), name='var_a')
    tf.summary.histogram('a_hist', a)
    b = tf.Variable(tf.truncated_normal([100, 1], mean=-0.5, stddev=1.0), name='var_b')
    tf.summary.histogram('b_hist', b)
    increase_b = tf.assign(b, b + 0.2)
    c = tf.add(a, b)
    tf.summary.histogram('c_hist', c)
    c_mean = tf.reduce_mean(c)
    tf.summary.scalar('c_mean', c_mean)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(log_dir+'add_example', sess.graph)


sess.run(tf.global_variables_initializer())
for step in range(500):
    sess.run([merged, increase_b])    # 每步改变一次 b 的值
    summary = sess.run(merged)
    writer.add_summary(summary, step)
writer.close()


#  在Anaconda命令行中输入 tensorboard --logdir="log_dir_path"(你保存到log路径)
#  http://localhost:6006/

# scalar 标量仪表盘，统计tensorflow中的标量（如：学习率、模型的总损失）随着迭代轮数的变化情况
# histogram 张量仪表盘，统计tensorflow中的张量随着迭代轮数的变化情况
# distributions 张量仪表盘，相较于HISTOGRAMS，用另一种直方图展示从tf.summary.histogram()函数记录的数据的规律
