import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

plotData = {'trainBatch': [], 'loss': []}  # 定义储存每次训练的结果
'''
1.线性拟合函数，通过定义目标函数和损失函数，再创建对话进行数据的拟合。
2.计算出但X的值为2时的结果，最后绘制出拟合函数和拟合过程中损失值变化的函数
3.报存检查点 生成Saver时调用tf.train.Saver(max_to_keep=1) 其中max_to_keep 为最大的保存点个数
4.调用saver。save（sess，dir.cpkt，global_strp=epoch) 会进行每次的报存检查点，其中路径的形式为dir-(数字).cpkt
5.加载每次的检查点,调用saver.restore(sess2,dir.cpk+str(检查点的序列))
'''
# 生成数据
train_x = np.linspace(-1, 1, 100)
train_y = 2 * train_x + np.random.rand(*train_x.shape) * .3  # 加入噪点
# 绘制原始数据分布图
plt.plot(train_x, train_y, 'ro', label='Original Data')
plt.legend()
plt.show()
# 创建模型
X = tf.placeholder('float')
Y = tf.placeholder('float')
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.zeros([1]), name='bias')
# 前向计算
Z = tf.multiply(X, W) + b
# 反向计算
loss = tf.reduce_mean(tf.square(Z - Y))  # 利用平方差表示损失函数
# 优化函数
learning_rate = 0.01  # 学习率
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)  # 最小化损失函数
# 训练模型
train_epochs = 20  # 训练次数
display_epoch = 4  # 训练显示次数
# 检查点方法保存模型
# saver = tf.train.Saver(max_to_keep=1)  # 生成saver,其中的max_to_keep 为检查点的最大个数
# 生成检查点的saver
saver = tf.train.Saver(max_to_keep=1)  # 其中最大的保存次数为1
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # 初始化运算图
    for epoch in range(train_epochs):
        for (x, y) in zip(train_x, train_y):
            sess.run(optimizer, feed_dict={X: x, Y: y})
        if epoch % display_epoch == 0:
            cost = sess.run(loss, feed_dict={X: train_x, Y: train_y})
            print('epoch:', epoch + 1, 'W=', sess.run(W), 'b=', sess.run(b), 'loss=', cost)
            if not (loss == 'NA'):
                plotData['trainBatch'].append(epoch)
                plotData['loss'].append(cost)
            # saver.save(sess, 'model/lineRegression.cpkt', global_step=epoch)  # 每一次保存模型
            saver.save(sess, 'model/lineRegressionModel.cpkt', global_step=epoch)  # 保存检查点
    print('Finish!')
    result = sess.run(Z, feed_dict={X: 2})  # 计算给定数据的结果
    print('The result of Model(W:{},b:{}) when x=2  is:{}'.format(sess.run(W), sess.run(b), result))
    # 绘制拟合图像
    plt.title('fit figure')
    plt.plot(train_x, train_y, 'ro', label='Original Data')
    plt.plot(train_x, sess.run(W) * train_x + sess.run(b), 'b-', label='Fit line')
    plt.legend()
    plt.show()
# 绘制拟合过程中误差变化曲线
plt.plot(plotData['trainBatch'], plotData['loss'], 'b--', label='Fit loss line')
plt.legend()
plt.show()

# 重启一个Session
load_epoch = 16
with tf.Session() as sess2:
    sess2.run(tf.global_variables_initializer())  # 初始化Session
    # saver.restore(sess2, 'model/lineRegression.cpkt-' + str(load_epoch))
    # 加载检查点的数据
    saver.restore(sess2, "model/lineRegression.cpkt-" + str(load_epoch))
    print('X=2,z={}'.format(sess2.run(Z, feed_dict={X: 2})))
#
