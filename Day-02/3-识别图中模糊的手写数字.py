import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pylab

mnist = input_data.read_data_sets('MINIST_DATA/', one_hot=True)
tf.reset_default_graph()  # 初始化图
# 定义占位符
x = tf.placeholder(tf.float32, [None, 784])  # MNIST 数据集
y = tf.placeholder(tf.float32, [None, 10])  # 数字0-9
# 定义模型
W = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.zeros([10]))
# 定义输出节点
pred = tf.nn.softmax(tf.matmul(x, W) + b)  # Softmax分类
# 定义反向传播结构
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))
tf.summary.scalar('loss-function', cost)  # 将损失以标量显示
# 定义参数
learning_rate = 0.01
# 使用梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
training_epochs = 25
batch_size = 100
display_step = 1
with tf.Session() as sess:
    # 初始化
    sess.run(tf.global_variables_initializer())
    merged_summary_op = tf.summary.merge_all()  # 合并所有的summary
    # 创建Summary_writer,用于写文件
    summary_writer = tf.summary.FileWriter('log/mnist_with_summaries', sess.graph)
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})
            avg_cost += c / total_batch
        # 显示训练中的详细信息
        if (epoch + 1) % display_step == 0:
            print('epoch:', epoch + 1, 'cost', avg_cost)
    print(" Finished!")
    # 测试Model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # 计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
