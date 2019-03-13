import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


v1 = tf.Variable([11.0, 16.3], name='v1')
v2 = tf.Variable(33.5, name='v2')

saver = tf.train.Saver()

ckpt_path = './ckpt/test-model.ckpt'
saver.restore(sess, ckpt_path + '-'+ str(1))
print("Model restored.")

print(sess.run(v1))
print(sess.run(v2))