# 代码先行

[English](./index.md) | 简体中文

模型通常采用以下训练步骤：

1. **定义模型架构**：指定神经网络的类型（例如，transformer），层数，每层的大小，以及学习率、批大小等超参数。
2. **准备和预处理数据**：收集大规模数据集，进行预处理，包括分词、去噪音，并将数据转换为适合模型处理的格式。
3. **初始化模型**：为模型中的权重和偏置设置初始值。
4. **前向传递**：将输入数据传递给模型以生成预测结果。模型逐层处理输入数据，生成输出。
5. **计算损失**：使用损失函数（例如分类任务中的交叉熵损失）将模型输出与目标输出进行比较。损失量化模型预测与实际目标之间的差距。
6. **反向传递（反向传播）**：通过应用链式法则，计算损失相对于每个模型参数（权重和偏置）的梯度。这一步将误差向网络的反向传播。
7. **更新权重**：使用优化算法（例如随机梯度下降，Adam）调整模型的权重，以最小化损失。优化器根据计算的梯度更新权重。
8. **遍历数据集**：对数据集进行多次迭代（训练轮次），重复前向传递、损失计算、反向传递和权重更新。这包括将数据分批传递给模型，不断调整权重。
9. **验证和微调**：定期在单独的验证数据集上评估模型性能，监控其表现以避免过拟合。微调超参数并进行必要的调整。
10. **模型准备就绪**：一旦训练收敛且模型在训练和验证数据集上表现良好，即可认为模型准备好部署。模型还可以进一步在特定任务或数据集上进行微调。

现在有数据如下:

| 输入 x | 结果 y |
| ------ | ------ |
| 1      | 3      |
| 2      | 5      |
| 3      | 7      |

求 x，y 之间的函数 y=f(x)。

假设对数据 x，生成数据 y 的函数为 y = ax + b，通过头两组数据的计算，我们可以得出 a = 2, b = 1, 带入第三组输入输出，确认 a 和 b 的数值计算无误。

然而，假如引入新数据 x = 4, y = 8, 我们看到由 y = 2x + 1 计算出的 y 为 9，并不等于 8，那么这时候我们可知假设函数的计算有误，那么我们需要调整计算过程，重新找到 y 与 x 之间的函数关系。

下面我们用代码来示例这个简单的过程：

```python
import torch

x = torch.ones(5)  # 输入
y = torch.zeros(3)  # 目标输出
w = torch.randn(5, 3, requires_grad=True)  # 初始参数
b = torch.randn(3, requires_grad=True)  # 初始参数

learning_rate = 0.01 # 学习率控制每次更新模型参数（权重和偏置）时根据估计误差改变模型参数的幅度。
optimizer = torch.optim.SGD([w, b], lr=learning_rate)  # 随机梯度下降（SGD）：使用损失函数相对于参数的梯度来更新参数。

steps = 0

while steps < 5:
    print(f"Training step {steps}: ")

    # 在前向传递之前将梯度清零
    optimizer.zero_grad()

    z = torch.matmul(x, w) + b  # 目标函数 z = w * x + b
    loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)  # 计算损失
    loss.backward()  # 反向传递

    print(w.grad)
    print(b.grad)

    # 更新权重
    optimizer.step()

    steps += 1
```

进一步阅读：

- [llm.c](https://github.com/karpathy/llm.c) 是一个用于语言建模的 transformer 模型的简单实现。Karpathy 对代码和背后的数学原理进行了详细解释。

- [PyTorch Tutorial](https://pytorch.org/tutorials/beginner/basics/intro.html)
