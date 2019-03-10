import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# 1. 使用 tf.Variable() 和 tf.get_variable() 创建变量

# 1.1
v1 = tf.Variable(initial_value=[1.0], name='v')
v2 = tf.Variable(initial_value=[2.0], name='v')
v3 = tf.Variable(initial_value=[1.0, 2.0], name='v')
print('v1', v1)
print('v2', v2)
print('v3', v3)

# 1.2
gv1 = tf.get_variable(name='gv', shape=[2, 3], initializer=tf.truncated_normal_initializer())
# # gv2 = tf.get_variable(name='gv', shape=[2,3], initializer=tf.truncated_normal_initializer())
print('gv1', gv1)

# 1.3 tf.Variable() 和 tf.get_variable() 同时创建变量，会自动处理。
var1 = tf.Variable(initial_value=[1.0], name='var', trainable=False)
var2 = tf.get_variable(name='var', shape=[2, 3])
var3 = tf.Variable(initial_value=[1.0], name='var')
# var4 = tf.get_variable(name='var', shape=[2,3])  # 报错
print('var1', var1)
print('var2', var2)
print('var3', var3)

# 1.4 使用 tf.placeholder() 创建占位符
ph1 = tf.placeholder(dtype=tf.float32, shape=[1, 3], name='ph')
ph2 = tf.placeholder(dtype=tf.float32, shape=[2, 3], name='ph')
print('ph1:', ph1)
print('ph2:', ph2)

# 1.5 获取全部的变量和 trainable 变量
all_vars = tf.global_variables()
for i in range(len(all_vars)):
    print(i, all_vars[i])

all_trainable_vars = tf.trainable_variables()
for i in range(len(all_trainable_vars)):
    print(i, all_trainable_vars[i])




