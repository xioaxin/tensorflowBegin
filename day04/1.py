import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 测试数据集
train_x =np.linspace(-1, 1,1000)
train_y = 2 * train_x + (np.random.randint(-1, 1,(1000,)) * 0.3)
# 绘图
plt.plot(train_x, train_y, 'ro')
plt.show()
