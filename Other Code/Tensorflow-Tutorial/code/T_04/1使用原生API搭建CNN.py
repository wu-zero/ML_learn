import warnings
warnings.filterwarnings('ignore')  # 不打印 warning

import tensorflow as tf

# 设置GPU按需增长
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

import numpy as np


# 1. 导入数据
# 用tensorflow 导入数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../data/MNIST_data', one_hot=True)
# 看看咱们样本的数量
print(mnist.test.labels.shape)
print(mnist.train.labels.shape)

# 构建网络
def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') # 默认 strides[0]=strides[3]=1, strides[1]为x方向步长，strides[2]为y方向步长

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # ksize 池化窗口 # strides 滑动步长


X_input = tf.placeholder(tf.float32, [None, 784])
y_input = tf.placeholder(tf.float32, [None, 10])

# 把X转为卷积所需要的形式
X = tf.reshape(X_input, [-1, 28, 28, 1])

# 第一层卷积
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(X, W_conv1) + b_conv1)
# 第一层池化
h_pool1 = max_pool_2x2(h_conv1)
# 第二层卷积
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# 第二层池化
h_pool2 = max_pool_2x2(h_conv2)
# 展平
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
# 全连接
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout: 输出的维度和h_fc1一样，只是随机部分值被值为零
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_pred = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

print('X_input:', X_input)
print('X:      ', X)
print('y_input:', y_input)
print('h_conv1:', h_conv1)
print('h_pool1:', h_pool1)
print('h_conv2:', h_conv2)
print('h_pool2:', h_pool2)
print('h_fc1:  ', h_fc1)
print('y_pred: ', y_pred)

# 3. 训练评估
# 损失函数
cross_entropy = -tf.reduce_sum(y_input * tf.log(y_pred))
# 优化函数
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 预测准确结果统计
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_input, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 定义了变量必须要初始化，或者下面形式
sess.run(tf.global_variables_initializer())

import time


time0 = time.time()
# 训练
for i in range(5000):
    X_batch, y_batch = mnist.train.next_batch(batch_size=100)
    cost, acc,  _ = sess.run([cross_entropy, accuracy, train_step], feed_dict={X_input: X_batch, y_input: y_batch, keep_prob: 0.5})
    if (i+1) % 500 == 0:
        # 分 100 个batch 迭代
        test_acc = 0.0
        test_cost = 0.0
        N = 100
        for j in range(N):
            X_batch, y_batch = mnist.test.next_batch(batch_size=100)
            _cost, _acc = sess.run([cross_entropy, accuracy], feed_dict={X_input: X_batch, y_input: y_batch, keep_prob: 1.0})
            test_acc += _acc
            test_cost += _cost
        print("step {}, train cost={:.6f}, acc={:.6f}; test cost={:.6f}, acc={:.6f}; pass {}s".format(i+1, cost, acc, test_cost/N, test_acc/N, time.time() - time0))
        time0 = time.time()

# 4. 查看中间层结果

# 数据
import matplotlib.pyplot as plt
img1 = mnist.train.images[0]
label1 = mnist.train.labels[0]
print(label1)
print('img_data shape =', img1.shape)  # 我们需要把它转为 28 * 28 的矩阵
img1 = img1.reshape([28, 28])
plt.imshow(img1, cmap='gray')
plt.axis('off')  # 不显示坐标轴
plt.show()


# 首先应该把 img1 转为正确的shape (None, 784)
X_img = img1.reshape([-1, 784])
y_img = mnist.train.labels[1].reshape([-1, 10])
# 看 Conv1 的结果，即 h_conv1
result = sess.run([h_conv1], feed_dict={X_input: X_img, keep_prob:1.0})[0]
print(result.shape)
print(type(result))

for i in range(32):
    show_img = result[:, :, :, i]
    show_img = show_img.reshape([28, 28])
    plt.subplot(4, 8, i + 1)
    plt.imshow(show_img, cmap='gray')
    plt.axis('off')
plt.show()