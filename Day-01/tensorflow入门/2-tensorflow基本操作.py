import tensorflow as tf

'''
tensorflow 的基本操作
'''
# 占位符进行计算
X = tf.placeholder('float')
Y = tf.placeholder('float')
Z = X + Y
with tf.Session() as sess:
    result = sess.run(Z, feed_dict={X: 1, Y: 2})
    print('The result of using placeHolder  is :{}'.format(result))

# 常量进行计算
a = tf.constant(1)
b = tf.constant(2)
with tf.Session() as sess:
    result = sess.run(a + b)
    print('The result of constant number operator is: {}'.format(result))
