import tensorflow as tf

'''
加载模型的参数，先创建一个saver再调用restore函数加载模型给sess，然后可以通过sess进行操作
'''
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.zeros([1]), name='bias')
saver = tf.train.Saver()  # 创建一个saver
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # 初始化对话
    saver.restore(sess, 'model/lineRegressionModel')  # 加载模型，此处一定得对应完整的名字
    print('The params of Model is W={},b={}'.format(sess.run(W), sess.run(b)))
