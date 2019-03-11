import numpy as np
import time
import tensorflow as tf

# 设置按需使用GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)

# 1. 导入数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../data/MNIST_data', one_hot=False)
# 看看咱们样本的数量
print(mnist.test.labels.shape)
print(mnist.train.labels.shape)

# 2. 构建网络
with tf.name_scope('inputs'):
    X_ = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.int64, [None])

X = tf.reshape(X_, [-1, 28, 28, 1])
h_conv1 = tf.layers.conv2d(X, filters=32, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu, name='conv1')
h_pool1 = tf.layers.max_pooling2d(h_conv1, pool_size=2, strides=2, padding='same', name='pool1')

h_conv2 = tf.layers.conv2d(h_pool1, filters=64, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu, name='conv2')
h_pool2 = tf.layers.max_pooling2d(h_conv2, pool_size=2, strides=2, padding='same', name='pool2')

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.layers.dense(h_pool2_flat, 1024, name='fc1', activation=tf.nn.relu)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
h_fc2 = tf.layers.dense(h_fc1_drop, units=10, name='fc2')
y_conv = h_fc2
print('Finished building network')

print(h_conv1)
print(h_pool1)
print(h_conv2)
print(h_pool2)

print(h_pool2_flat)
print(h_fc1)
print(h_fc2)


# 3. 训练和评估
# 损失函数
# cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.cast(y_, dtype=tf.int32), logits=y_conv))
# 优化函数
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 预测准确结果统计
correct_prediction = tf.equal(tf.argmax(y_conv, 1), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 定义了变量必须要初始化
sess.run(tf.global_variables_initializer())


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

