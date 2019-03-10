# 2. 使用 tf.name_scope() 和 tf.variable_scope() 管理命名空间
import warnings
warnings.filterwarnings('ignore')  # 不打印 warning

import tensorflow as tf

# 设置GPU按需增长
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# 2.1 使用 tf.name_scope()
with tf.name_scope('nsl') as ns:
    ns_v1 = tf.Variable(initial_value=[1.0], name='v')
    ns_gv1 = tf.get_variable(name='v', shape=[2,3])
    ns_v2 = tf.Variable(initial_value=[1.0], name='v')

print('ns_v1', ns_v1)
print('ns_gv1', ns_gv1)
print('ns_v2', ns_v2)

# 2.2 使用 tf.variable_scope()
with tf.variable_scope('vs1') as vs:
    vs_v1 = tf.Variable(initial_value=[1.0], name='v')
    vs_gv1 = tf.get_variable(name='v', shape=[2,3])
    # vs_gv2 = tf.get_variable(name='v', shape=[1,3])  # ValueError: Variable vs1/v already exists

print('vs_v1', vs_v1)
print('vs_gv1', vs_gv1)

# 2.3 tf.variable_scope() 中设置 reuse=True
with tf.variable_scope('vs1') as vs:
    vs_gv2 = tf.get_variable(name='v2', shape=[2, 3])
    # vs_gv3 = tf.get_variable(name='v', shape=[2,3])  # ValueError: Variable vs1/v already exists
print('vs_gv2', vs_gv2)

with tf.variable_scope('vs1', reuse=True) as vs:
    vs_gv3 = tf.get_variable(name='v', shape=[2, 3])

print('vs_gv3', vs_gv3)
print('vs_gv3 is vs_gv1: ', vs_gv3 is vs_gv1)