import tensorflow as tf
import numpy as np

# 张量的普通加、减、乘、除法
a = tf.constant(1)
b = tf.constant(4)
tf.reset_default_graph
with tf.Session() as sess:
    # sess.run(tf.global_variables_initializer())
    print('相加为:{}'.format(sess.run(a + b)))
    print('相减为:{}'.format(sess.run(a - b)))
    print('相乘为:{}'.format(sess.run((a * b))))
    print('相除为:{}'.format(sess.run(a / b)))

# 调用函数的计算
x = tf.constant(5)
y = tf.constant(6)
print('相加为:{}'.format(tf.add(x, y)))
print('相减为:{}'.format(tf.subtract(x, y)))
print('相乘为:{}'.format(tf.multiply(x, y)))
print('相除为:{}'.format(tf.div(x, y)))
