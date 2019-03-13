import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

v1 = tf.Variable([1.0, 2.3], name='v1')
v2 = tf.Variable(55.5, name='v2')

init_op = tf.global_variables_initializer()

saver = tf.train.Saver()
ckpt_path = './ckpt/test-model.ckpt'
sess.run(init_op)
save_path = saver.save(sess,ckpt_path, global_step=1)
print("Model saved in file: %s" % save_path)
