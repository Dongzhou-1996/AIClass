import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.metrics import accuracy_score
from scipy.stats import mode

# Step 1: 定义 K-Means函数
def kmeans(X, k, max_iters=100, tol=1e-4):
    """
    自实现 K-Means 聚类算法
    :param X: 数据集 (numpy array)
    :param k: 聚类数
    :param max_iters: 最大迭代次数
    :param tol: 允许的误差变化
    :return: labels (样本所属的簇), centers (最终聚类中心)
    """
    np.random.seed(42)  # 固定随机种子
    n_samples, n_features = X.shape

    # **Step 1.1: 随机选择 k 个初始中心**
    initial_indices = np.random.choice(n_samples, k, replace=False)
    centers = X[initial_indices]

    for _ in range(max_iters):
        # **Step 2: 计算样本到各个中心的距离**
        distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)  # 计算欧几里得距离
        labels = np.argmin(distances, axis=1)  # 每个点分配到最近的簇

        # **Step 3: 计算新的簇中心**
        new_centers = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        # **Step 4: 检查收敛条件**
        if np.linalg.norm(new_centers - centers) < tol:
            break

        centers = new_centers

    return labels, centers

# **Step 2: 加载 Iris 数据集**
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target  # 真实标签

# **Step 3: 运行 K-Means**
k = 3  # 设定聚类数
X = iris_df.drop(columns=['target']).values  # 取出特征数据

labels, centers = kmeans(X, k)  # 运行自定义 K-Means
iris_df['cluster_labels'] = labels  # 记录聚类标签

# **Step 4: 计算聚类准确率**
mapped_labels = np.zeros_like(labels)
for i in range(k):
    mask = (labels == i)
    mapped_labels[mask] = mode(iris_df['target'][mask])[0]  # 重新映射类别

accuracy = accuracy_score(iris_df['target'], mapped_labels)
print(f"聚类准确率: {accuracy:.2f}")

# **Step 5: 进行可视化**
plt.figure(figsize=(10, 6))
scatter = plt.scatter(iris_df['petal length (cm)'], iris_df['petal width (cm)'],
                      c=iris_df['cluster_labels'], cmap='viridis', marker='o', edgecolor='k')

# 添加聚类中心
plt.scatter(centers[:, 2], centers[:, 3], c='red', marker='*', s=200, label='Cluster Centers')

# 添加图例
plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=8, label='Cluster 0'),
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Cluster 1'),
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=8, label='Cluster 2'),
                    plt.Line2D([0], [0], marker='*', color='red', label='Cluster Centers', markersize=10)],
           loc='upper left')

# 设置图表标题和轴标签
plt.title('K-means Clustering of Iris Dataset (Petal Length vs Petal Width)')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')

# 显示图表
plt.show()
