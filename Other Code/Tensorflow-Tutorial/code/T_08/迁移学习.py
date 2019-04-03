import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# 导入v1
v1 = tf.Variable([11.0, 16.3], name='v1')
saver = tf.train.Saver()
ckpt_path = '../T_07/ckpt/test-model.ckpt'
saver.restore(sess, ckpt_path + '-' + str(1))
print(sess.run(v1))

# 定义新v3
v3 = tf.Variable(23, name='v3', dtype=tf.int32)
init_new = tf.variables_initializer([v3])
sess.run(init_new)


print(sess.run(v1))
print(sess.run(v3))