# 常见术语

[English](./index.md) | 简体中文

**梯度**：梯度衡量当你稍微改变输入时，函数输出的变化量。它是相对于所有输入变量的偏导数向量。梯度用于优化算法中，例如梯度下降，通过迭代调整模型参数来最小化损失函数，从而减少误差。

**损失函数**：损失函数衡量机器学习模型的预测与实际数据的匹配程度。它量化了预测值与真实值之间的差异。训练模型的目标是最小化损失函数，从而提高模型的准确性。

**预训练**：预训练是在针对特定任务进行微调之前，先在大型数据集上训练模型的过程。这个初始训练阶段使模型能够从广泛的数据中学习一般特征和模式，建立可以转移到更专业任务的知识基础。预训练在深度学习模型（如 transformer）中尤为重要，例如 BERT、GPT 等模型首先在海量文本语料库上训练以学习语言表示，然后再针对问答、情感分析或文本生成等特定应用进行微调。这种方法显著减少了特定任务所需的数据量，并提高了下游任务的性能。

**微调**：微调是对预训练的机器学习模型进行小幅调整以适应特定任务或数据集的过程。这包括在新的、通常较小的数据集上再训练模型几个周期，使其在保留初始训练阶段获得的一般知识的同时，学习新数据的细微差别。微调常用于迁移学习，以提高模型在特定任务上的表现。

**张量**：张量是一个多维数组，它将标量（0 维）、向量（1 维）和矩阵（2 维）的概念推广到更高维度。张量用于表示各种形式的数据，例如神经网络中的输入数据、权重和激活。它们是机器学习框架（如 TensorFlow 和 PyTorch）中的基本数据结构，能够高效地计算和操作大规模数据。

**超参数**：超参数是用于构建和训练模型的配置设置。与在训练过程中学习的模型参数不同，超参数是在训练过程开始之前设置的。示例包括学习率、批大小、训练周期数以及神经网络的架构（如层数和每层的单元数）。正确调整超参数对于优化模型性能和获得最佳结果至关重要。

**优化器**：在机器学习中，特别是在神经网络的背景下，优化器是一种用于调整模型参数（权重和偏置）以最小化损失函数的算法或方法。优化器的目标是找到一组参数，使模型在给定任务上的表现最佳。

**强化学习**：强化学习是一种机器学习类型，其中智能体通过在环境中采取行动来学习决策，以最大化某种累积奖励。智能体通过试错学习，以奖励或惩罚的形式接收反馈。与监督学习不同，智能体不会被告知应该采取哪些行动，而是必须通过探索环境来发现哪些行动能够产生最高的奖励。关键组成部分包括智能体（决策者）、环境（智能体交互的对象）、行动（智能体可以做什么）、状态（智能体所处的情况）和奖励（来自环境的反馈）。强化学习应用于各种领域，如游戏、机器人技术、自动驾驶车辆和推荐系统。
