from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import tensorflow as tf

saveDir = 'model/'
print_tensors_in_checkpoint_file(saveDir + 'lineRegressionModel.cpkt', None, True)
# 创建变量
W = tf.Variable(1.0, name="weight")
b = tf.Variable(2.0, name='bias')
# 创建Saver
saver = tf.train.Saver({'wight:': W, "bias": b})
# 创建会话
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # 初始化会话
    saver.save(sess, saveDir + 'lineRegressionModel.cpkt')
print_tensors_in_checkpoint_file(saveDir + 'lineRegressionModel.cpkt', None, True)
