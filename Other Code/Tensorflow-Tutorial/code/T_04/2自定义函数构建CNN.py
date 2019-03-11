import numpy as np
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)

# 1. 导入数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../data/MNIST_data', one_hot=True)
# 看看咱们样本的数量
print(mnist.test.labels.shape)
print(mnist.train.labels.shape)

# 2. 构建网络

# 定义卷积层
def conv2d(x, filter_shape, strides_x, strides_y, padding, name):
    assert padding in ['SAME', 'VALID']
    strides = [1, strides_x, strides_y, 1]
    with tf.variable_scope(name):
        W_conv = tf.get_variable('W', shape=filter_shape)
        b_conv = tf.get_variable('b', shape=[filter_shape[-1]])
        h_conv = tf.nn.conv2d(x, W_conv, strides=strides, padding=padding)
        h_conv_relu = tf.nn.relu(h_conv + b_conv)
    return h_conv_relu

def max_pooling(x, k_height, k_width, strides_x, strides_y, padding='SAME'):
    ksize = [1, k_height, k_width, 1]
    strides = [1, strides_x, strides_y, 1]
    h_pool = tf.nn.max_pool(x, ksize, strides, padding)
    return h_pool

def dropout(x, keep_prob, name=None):
    return tf.nn.dropout(x, keep_prob, name=name)

def fc(x, in_size, out_size, name, activation=None):
    if activation is not None:
        assert activation in ['relu', 'sigmoid', 'tanh'], 'Wrong activate function'
    with tf.variable_scope(name):
        w = tf.get_variable('w', shape=[in_size, out_size], dtype=tf.float32)
        b = tf.get_variable('b', shape=[out_size], dtype=tf.float32)
        h_fc = tf.nn.xw_plus_b(x, w, b)
        if activation == 'relu':
            return tf.nn.relu(h_fc)
        elif activation == 'tanh':
            return tf.nn.tanh(h_fc)
        elif activation == 'sigmoid':
            return tf.nn.sigmoid(h_fc)
        else:
            return h_fc


with tf.name_scope('inputs'):
    X_ = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])

X = tf.reshape(X_, [-1, 28, 28, 1])
h_conv1 = conv2d(X, [5, 5, 1, 32], 1, 1, 'SAME', 'conv1')
h_pool1 = max_pooling(h_conv1, 2, 2, 2, 2)

h_conv2 = conv2d(h_pool1, [5, 5, 32, 64], 1, 1, 'SAME', 'conv2')
h_pool2 = max_pooling(h_conv2, 2, 2, 2, 2)

# flatten
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = fc(h_pool2_flat, 7*7*64, 1024, 'fc1', 'relu')

# dropout: 输出的维度和h_fc1一样，只是随机部分值被值为零
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
h_fc2 = fc(h_fc1_drop, 1024, 10, 'fc2')
y_conv = tf.nn.softmax(h_fc2)
print('Finished building network.')


#  3.训练和评估

# 损失函数
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
# 优化函数
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 预测准确结果统计
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 定义了变量必须要初始化
sess.run(tf.global_variables_initializer())
# 或者某个变量单独初始化 如：
# x.initializer.run()


import time
time0 = time.time()
# 训练
for i in range(5000):
    X_batch, y_batch = mnist.train.next_batch(batch_size=100)
    cost, acc, _ = sess.run([cross_entropy, accuracy, train_step],
                            feed_dict={X_: X_batch, y_: y_batch, keep_prob: 0.5})
    # 显示训练过程结果
    if (i + 1) % 500 == 0:
        # 分 100 个batch 迭代
        test_acc = 0.0
        test_cost = 0.0
        N = 100
        for j in range(N):
            X_batch, y_batch = mnist.test.next_batch(batch_size=100)
            _cost, _acc = sess.run([cross_entropy, accuracy],
                                   feed_dict={X_: X_batch, y_: y_batch, keep_prob: 1.0})
            test_acc += _acc
            test_cost += _cost
        print("step {}, train cost={:.6f}, acc={:.6f}; test cost={:.6f}, acc={:.6f}; pass {}s".format(i + 1, cost, acc,
                                                                                                      test_cost / N,
                                                                                                      test_acc / N,
                                                                                             time.time() - time0))
        time0 = time.time()


