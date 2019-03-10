import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# 把变量添加到一个collection中
v1 = tf.Variable([1, 2, 3], name='v1')
v2 = tf.Variable([2], name='v2')
v3 = tf.get_variable(name='v3', shape=(2, 3))

tf.add_to_collection('coll', v1)
tf.add_to_collection('coll', v3)

colls = tf.get_collection(key='coll')
print('vars in coll:', colls)

op1 = tf.add(v1, v2, name='add_op')
tf.add_to_collection('coll', op1)
colls = tf.get_collection(key='coll')
print('vars in coll:', colls)

with tf.variable_scope('model'):
    v4 = tf.get_variable('v4', shape=[3, 4])
    v5 = tf.Variable([1, 2, 3], name='v5')

tf.add_to_collection('coll', v5)
coll_vars = tf.get_collection(key='coll', scope='model')
print('vars in coll with scope=model:', coll_vars)


