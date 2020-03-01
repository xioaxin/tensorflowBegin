import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

train_x = np.linspace(-1, 1, 100)
train_y = 2 * train_x + np.random.rand(*train_x.shape) * 0.3
plt.plot(train_x, train_y, 'ro', label='Original Data')
plt.legend()
plt.show()
X = tf.placeholder('float')
Y = tf.placeholder('float')
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.zeros([1]), name='bias')
# 前向结构
Z = tf.multiply(X, W) + b
# 将预测值以直方图显示
tf.summary.histogram('z', Z)
cost = tf.reduce_mean(tf.square(Y - Z))
tf.summary.scalar('loss-function', cost)  # 将损失以标量显示
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# 初始化变量
init = tf.global_variables_initializer()
train_epochs = 20
display_epoch = 2
with tf.Session() as sess:
    sess.run(init)
    merged_summary_op = tf.summary.merge_all()  # 合并所有的summary
    # 创建Summary_writer,用于写文件
    summary_writer = tf.summary.FileWriter('log/mnist_with_summaries', sess.graph)
    for epoch in range(train_epochs):
        for (x, y) in zip(train_x, train_y):
            sess.run(optimizer, feed_dict={X: x, Y: y})
            # 生成summary
            summary_str = sess.run(merged_summary_op, feed_dict={X: x, Y: y})
            summary_writer.add_summary(summary_str, epoch)  # 将summary写入文件
        if epoch % display_epoch == 0:
            loss = sess.run(cost, feed_dict={X: train_x, Y: train_y})
            print('epoch:', epoch + 1, 'W=', sess.run(W), "b=", sess.run(b), 'loss=', loss)
