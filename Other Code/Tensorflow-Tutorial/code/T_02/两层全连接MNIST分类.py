from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# 1. 导入数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./data/MNIST_data', one_hot=False, source_url='http://yann.lecun.com/exdb/mnist/')

print('training data shape ', mnist.train.images.shape)
print('training label shape ', mnist.train.labels.shape)


# 2. 构建网络
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  # 从截断的正态分布中输出随机值。
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)  # 生成常量
    return tf.Variable(initial)


X_input = tf.placeholder(tf.float32, [None, 784])  # 此函数可以理解为形参，用于定义过程，在执行的时候再赋具体的值
y_input = tf.placeholder(tf.int64, [None])

W_fc1 = weight_variable([784, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(X_input, W_fc1) + b_fc1)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

logits = tf.matmul(h_fc1, W_fc2 + b_fc2)
print(logits)

# 3. 训练和评估

# 损失函数   tf.nn.sigmoid_cross_entropy_with_logits
#           tf.nn.softmax_cross_entropy_with_logits
#           tf.nn.sparse_softmax_cross_entropy_with_logits
cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_input, logits=logits))

# 优化函数
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 预测结果评估
correct_prediction = tf.equal(tf.argmax(logits, 1), y_input)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # tf.cast() 数据类型转换

# 将所有全局变量的初始化器汇总，并对其进行初始化。
sess.run(tf.global_variables_initializer())

for i in range(5000):
    X_batch, y_batch = mnist.train.next_batch(batch_size=100)  # 每一步迭代，我们都会加载100个训练样本
    cost, acc, _ = sess.run([cross_entropy, accuracy, train_step], feed_dict={X_input: X_batch, y_input: y_batch})
    if (i + 1) % 500 == 0:
        test_cost, test_acc = sess.run([cross_entropy, accuracy],
                                       feed_dict={X_input: mnist.test.images, y_input: mnist.test.labels})
        print("step {}, train cost={:.6f}, acc={:.6f}; test cost={:.6f}, acc={:.6f}".format(i + 1, cost, acc, test_cost,
                                                                                            test_acc))
