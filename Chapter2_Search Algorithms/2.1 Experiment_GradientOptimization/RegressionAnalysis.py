import numpy as np
import matplotlib.pyplot as plt

# 生成一些随机数据，实际使用中可以导入一些实际观测得到的数据列表
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 初始化参数
w = np.random.randn(1)
b = np.random.randn(1)

# 学习率
learning_rate = 0.1

# 迭代次数
n_iterations = 1000

# 梯度下降算法
for iteration in range(n_iterations):
    # 计算预测值
    y_pred = w * X + b

    # 计算损失函数（均方误差）
    loss = np.mean((y_pred - y) ** 2)

    # 计算梯度
    dw = np.mean(2 * (y_pred - y) * X)
    db = np.mean(2 * (y_pred - y))

    # 更新参数
    w = w - learning_rate * dw
    b = b - learning_rate * db

    # 打印损失函数值
    if iteration % 100 == 0:
        print(f"Iteration {iteration}, Loss: {loss}")

# 打印最终参数
print(f"Final parameters: w = {w}, b = {b}")

# 绘制结果
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, w * X + b, color='red', label='Fitted line')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
