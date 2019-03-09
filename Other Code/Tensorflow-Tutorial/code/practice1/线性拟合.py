import tensorflow as tf
import numpy as np

# 设置tensorflow对GPU的使用按需分配
# 如果直接使用 sess = tf.Session() 的话会默认占用全部的 GPU 资源
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# 1.准备数据：使用 NumPy 生成假数据(phony data), 总共 100 个点.
X = np.float32(np.random.rand(100))  # 随机输入
X = np.sort(X)
y = np.dot(0.200, X**2) + 0.300 + np.random.randn(100) * 0.01  # 加一点点噪声
X = X.reshape([-1, 1])
y = y.reshape([-1, 1])
print('X.shape=', X.shape)
print('y.shape=', y.shape)

# 2.构造一个线性模型
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1, 1], -1.0, 1.0))
y_pre = tf.matmul(X**2, W) + b

# 3.求解模型
loss = tf.reduce_mean(tf.square(y_pre - y))         # 设置损失函数：误差的均方差
optimizer = tf.train.GradientDescentOptimizer(0.5)   # 选择梯度下降的方法
train = optimizer.minimize(loss)                     # 迭代的目标：最小化损失函数


############################################################
# 以下是用 tf 来解决上面的任务
# 1.初始化变量：tf 的必备步骤，主要声明了变量，就必须初始化才能用
init = tf.global_variables_initializer()
sess.run(init)

# 2.迭代，反复执行上面的最小化损失函数这一操作（train op）,拟合平面
for step in range(0, 201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))

# 得到最佳拟合结果 W: [[0.100  0.200]], b: [0.300]



import matplotlib.pyplot as plt


_y_pre = sess.run(y_pre)
print('y_pre:', _y_pre.reshape([-1])[:20])   # 预测值
print('y_true:', y.reshape([-1])[:20]) # 真实值

plt.plot(X, y, 'b.')
plt.plot(X, _y_pre, 'r-')
plt.show()