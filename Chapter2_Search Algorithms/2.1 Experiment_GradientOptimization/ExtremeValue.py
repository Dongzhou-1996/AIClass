import numpy as np
import matplotlib.pyplot as plt


# 定义 Rastrigin 函数
def rastrigin(x):
    return 10 + x ** 2 - 10 * np.cos(2 * np.pi * x)


# 定义 Rastrigin 函数的梯度
# def grad_rastrigin(x):
# 前向差分逼近实际梯度   return 2 * x + 10 * 2 * np.pi * np.sin(2 * np.pi * x)
def compute_gradient(x):
    h = 1e-5  # 计算梯度的步长
    gradient = np.zeros_like(x)
    x_plus_h = np.copy(x) + h
    x_minus_h = np.copy(x) - h
    gradient = (rastrigin(x_plus_h) - rastrigin(x_minus_h)) / (2 * h)
    return gradient


# 初始化参数
x = np.random.uniform(-5, 5)  # 随机初始化 x 在 [-5, 5] 范围内

# 学习率
learning_rate = 0.01

# 最大迭代次数
n_iterations = 1000

# 存储每次迭代的 x 和 f(x) 值
x_history = []
f_history = []

# 随机梯度下降
for iteration in range(n_iterations):
    # 计算梯度
    gradient = compute_gradient(x)

    # 更新参数
    x = x - learning_rate * gradient

    # 存储历史值
    x_history.append(x)
    f_history.append(rastrigin(x))

    # 打印每次迭代的结果
    if iteration % 100 == 0:
        print(f"Iteration {iteration + 1}: x = {x}, f(x) = {rastrigin(x)}")

# 输出最终结果
print(f"Optimized x: {x}")
print(f"Minimum value of f(x): {rastrigin(x)}")

# 绘制结果
x_values = np.linspace(-5, 5, 1000)
y_values = rastrigin(x_values)

plt.plot(x_values, y_values, label='Rastrigin Function')
plt.scatter(x_history, f_history, color='red', label='SGD steps', alpha=0.5)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.show()
