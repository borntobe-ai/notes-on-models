# 推荐模型

[English](./index.md) | 简体中文

像Youtube, Netflix, TikTok等平台都通过推荐算法向用户推荐用户可能感兴趣的内容而受到欢迎。简单来说，模型就是学习数据背后的隐藏数学关系，Transformer架构主要用于处理顺序数据，而对推荐内容的数据标注通常是用户和内容之间的矩阵关联数据。

推荐采用的反馈数据的标注可以是显式的也可以是隐式的。
显式：用户直接标注对内容的偏好，例如评分或者评论

| 用户        | 电影1   | 电影2   | 电影3   | 电影4   | 电影5   |
| ----------- | -------- | -------- | -------- | -------- | -------- |
| 123456789   | 3        | 3        | 3        | 3        | 3        |
| 325469787   | 5        | 3        | 3        | 3        | 3        |
| 258649112   | 7        | 3        | 3        | 3        | 3        |

隐式：通过用户的观看时长或者点击，购买记录等等来表示用户的喜好

| 用户           | 内容 | 观看时长 |
| ----------- | --------- | -------- |
| 73371537119921 | 10007483 | 44652            |
| 51908012208654 | 10017079 | 121420           |
| 22936336127039 | 10051012 | 47744            |
| 58749733749324 | 10051012 | 32109            |
| 11736988012551 | 10067685 | 10512            |

这里我们简单介绍一下对隐式反馈进行计算的矩阵分解模型，使用[加权交替最小二乘算法](http://yifanhu.net/PUB/cf.pdf)（Weighted Alternating Least Squares, WALS）。

加权交替最小二乘算法是一种用于协同过滤的优化方法，特别适合处理隐式反馈数据。与显式反馈不同，隐式反馈只能表明用户与内容的交互，但不能直接反映用户的偏好程度。WALS算法通过以下步骤解决这个问题：

1. 将用户-物品交互矩阵转换为置信度矩阵，通常使用公式：置信度 = 1 + α * 观察值（其中α是超参数）
2. 交替优化用户因子矩阵和物品因子矩阵：
   - 固定物品因子，求解用户因子
   - 固定用户因子，求解物品因子
3. 每次优化都是一个加权最小二乘问题，通过求解线性方程组来完成

这种方法的优势在于它能有效处理大规模稀疏数据，并且通过加权机制区分观察到的交互和未观察到的交互，从而提高推荐质量。以下是代码示例：

```python
import numpy as np
from scipy.sparse import coo_matrix
import pandas as pd

class ImplicitMF:
    def __init__(self, n_factors=50, regularization=0.01, alpha=40, iterations=10):
        self.n_factors = n_factors
        self.regularization = regularization
        self.alpha = alpha
        self.iterations = iterations
    
    def fit(self, user_item_matrix):
        # 将观察到的交互转换为置信度
        confidence = 1 + self.alpha * user_item_matrix.toarray()
        
        n_users, n_items = user_item_matrix.shape
        
        # 初始化用户和物品因子矩阵
        self.user_factors = np.random.normal(size=(n_users, self.n_factors))
        self.item_factors = np.random.normal(size=(n_items, self.n_factors))
        
        # 加权交替最小二乘算法
        for _ in range(self.iterations):
            # 更新用户因子
            for u in range(n_users):
                # 创建一个对角矩阵Cu，对角线上的值是用户u对所有物品的置信度
                # confidence[u]是用户u对所有物品的置信度向量
                # flatten()将多维数组转为一维
                # np.diag()创建一个对角矩阵，对角线上的值来自输入向量
                Cu = np.diag(confidence[u].flatten())
                # 这行代码计算ALS算法中的系数矩阵A
                # self.item_factors.T 是物品因子矩阵的转置
                # Cu 是用户u对所有物品的置信度对角矩阵
                # self.item_factors 是物品因子矩阵
                # self.regularization * np.eye(self.n_factors) 添加正则化项，防止过拟合
                # 整个表达式计算出用于求解用户因子的线性方程组的系数矩阵
                A = self.item_factors.T @ Cu @ self.item_factors + self.regularization * np.eye(self.n_factors)
                # 这行代码计算ALS算法中的右侧向量b
                # self.item_factors.T 是物品因子矩阵的转置
                # Cu 是用户u对所有物品的置信度对角矩阵
                # (user_item_matrix[u].toarray().flatten() > 0) 创建一个布尔向量，表示用户u与哪些物品有交互
                # 整个表达式计算出用于求解用户因子的线性方程组的右侧向量
                b = self.item_factors.T @ Cu @ (user_item_matrix[u].toarray().flatten() > 0)
                # 这行代码使用线性代数求解方程组Ax=b，计算用户u的因子向量
                # np.linalg.solve函数求解线性方程组，其中A是系数矩阵，b是右侧向量
                # 求解结果是用户u的潜在因子，代表用户在潜在空间中的表示
                # 这些因子将用于与物品因子相乘来预测用户对物品的偏好
                self.user_factors[u] = np.linalg.solve(A, b)
            
            # 更新物品因子
            for i in range(n_items):
                # 创建一个对角矩阵Ci，对角线上的值是所有用户对物品i的置信度
                # confidence[:, i]是所有用户对物品i的置信度向量
                # flatten()将多维数组转为一维
                # np.diag()创建一个对角矩阵，对角线上的值来自输入向量
                Ci = np.diag(confidence[:, i].flatten())
                # 这行代码计算ALS算法中的系数矩阵A
                # self.user_factors.T 是用户因子矩阵的转置
                # Ci 是所有用户对物品i的置信度对角矩阵
                # self.user_factors 是用户因子矩阵
                # self.regularization * np.eye(self.n_factors) 添加正则化项，防止过拟合
                # 整个表达式计算出用于求解物品因子的线性方程组的系数矩阵
                A = self.user_factors.T @ Ci @ self.user_factors + self.regularization * np.eye(self.n_factors)
                # 这行代码计算ALS算法中的右侧向量b
                # self.user_factors.T 是用户因子矩阵的转置
                # Ci 是所有用户对物品i的置信度对角矩阵
                # (user_item_matrix[:, i].toarray().flatten() > 0) 创建一个布尔向量，表示哪些用户与物品i有交互
                # 整个表达式计算出用于求解物品因子的线性方程组的右侧向量
                b = self.user_factors.T @ Ci @ (user_item_matrix[:, i].toarray().flatten() > 0)
                # 这行代码使用线性代数求解方程组Ax=b，计算物品i的因子向量
                # np.linalg.solve函数求解线性方程组，其中A是系数矩阵，b是右侧向量
                # 求解结果是物品i的潜在因子，代表物品在潜在空间中的表示
                # 这些因子将用于与用户因子相乘来预测用户对物品的偏好
                self.item_factors[i] = np.linalg.solve(A, b)
    
    def recommend(self, user_id, n_items=10):
        # 计算用户对所有物品的预测评分
        # 用户因子和物品因子相乘代表在潜在空间中的相似度或兼容性
        # 用户因子向量表示用户在潜在特征空间中的偏好
        # 物品因子向量表示物品在相同潜在特征空间中的属性
        # 两者的点积（内积）计算它们在潜在空间中的匹配程度
        # 点积值越高，表示用户对该物品的偏好越强，即预测评分越高
        scores = self.user_factors[user_id] @ self.item_factors.T
        # 返回评分最高的n个物品
        return np.argsort(scores)[::-1][:n_items]

# 示例使用
# 构造示例数据
users = [0, 0, 1, 1, 2, 2]
items = [0, 1, 0, 2, 1, 2]
watch_time = [120, 80, 95, 150, 200, 45]  # 观看时长作为隐式反馈

# 创建用户-物品矩阵
user_item_matrix = coo_matrix((watch_time, (users, items)), shape=(3, 3)).tocsr()

# 训练模型
model = ImplicitMF(n_factors=10, iterations=20)
model.fit(user_item_matrix)

# 为用户0推荐物品
recommendations = model.recommend(user_id=0, n_items=2)
print(f"为用户0推荐的物品: {recommendations}")
```
