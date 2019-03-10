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
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
# 优化函数
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 预测准确结果统计
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 如果一次性来做测试的话，可能占用的显存会比较多，所以测试的时候也可以设置较小的batch来看准确率
test_acc_sum = tf.Variable(0.0)
batch_acc = tf.placeholder(tf.float32)
new_test_acc_sum = tf.add(test_acc_sum, batch_acc)
update = tf.assign(test_acc_sum, new_test_acc_sum)  # 赋值

# 定义了变量必须要初始化，或者下面形式
sess.run(tf.global_variables_initializer())
# 或者某个变量单独初始化 如：
# x.initializer.run()

# 训练
for i in range(5000):
    X_batch, y_batch = mnist.train.next_batch(batch_size=50)
    if i % 500 == 0:
        train_accuracy = accuracy.eval(feed_dict={X_: X_batch, y_: y_batch, keep_prob: 1.0})  # .eval 启动计算
        print("step %d, training acc %g" % (i, train_accuracy))
    train_step.run(feed_dict={X_: X_batch, y_: y_batch, keep_prob: 0.5})

# 全部训练完了再做测试，batch_size=100
for i in range(100):
    X_batch, y_batch = mnist.test.next_batch(batch_size=100)
    test_acc = accuracy.eval(feed_dict={X_: X_batch, y_: y_batch, keep_prob: 1.0})
    update.eval(feed_dict={batch_acc: test_acc})
    if (i+1) % 20 == 0:
        print("testing step %d, test_acc_sum %g" % (i+1, test_acc_sum.eval()))
print("test_accuracy %g" % (test_acc_sum.eval() / 100.0))