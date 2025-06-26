# Math Theory

English | [简体中文](./index_zh-CN.md)

## What is a Neural Network?

A neural network is a mathematical model inspired by biological neurons. At its core, it's a collection of interconnected nodes (neurons) that process information through weighted connections.
**Basic Structure**

- Input Layer: Receives data (numbers representing features)
- Hidden Layers: Process information through mathematical transformations
- Output Layer: Produces final predictions or classifications

**Mathematical Operations**
Each neuron performs two key operations:

1. Linear Combination: Multiplies inputs by weights and adds a bias: `z = w₁x₁ + w₂x₂ + ... + b`
2. Non-linear Activation: Applies a function like ReLU or sigmoid: `output = f(z)`

**Information Flow**
Data flows forward through the network, with each layer transforming the input into increasingly abstract representations. The network learns by adjusting weights to minimize prediction errors.

## Universal Function Approximation

Neural networks possess a remarkable mathematical property: they can approximate any continuous function to arbitrary precision. This is known as the Universal Approximation Theorem.

A neural network with just one hidden layer containing enough neurons can approximate any continuous function on a compact set. This works because:

**Basis Function Decomposition**
Each hidden neuron acts as a basis function - a simple building block. By combining enough of these blocks with appropriate weights, we can construct arbitrarily complex shapes and patterns, similar to how Fourier series use sine/cosine waves to represent any periodic function.

**Non-linear Activation Functions Are Key**
Without non-linearity, neural networks would only compute linear transformations. Activation functions like ReLU, sigmoid, or tanh introduce the bends and curves necessary to approximate complex, non-linear relationships.

**Weight Space Density**
The space of possible weight configurations is dense in the space of continuous functions. This means for any target function, there exists a set of weights that makes the network arbitrarily close to that function.

**Practical Implications**
While the theorem guarantees approximation capability, it doesn't specify:

- How many neurons are needed (could be exponentially large)
- How to find the right weights (training is still an optimization challenge)
- Whether the network will generalize beyond training data

This mathematical guarantee explains why neural networks are so powerful: given enough capacity and the right training, they can theoretically learn any pattern that can be expressed as a continuous function.

## Mathematical Fields

To fully understand neural networks, you'll need knowledge from several mathematical areas:

**Linear Algebra** - Essential for understanding matrix operations, vector spaces, and transformations that form the computational backbone of neural networks.

**Calculus** - Critical for backpropagation, gradient computation, and optimization. You'll need partial derivatives, chain rule, and multivariable calculus.

**Probability and Statistics** - Important for understanding loss functions, regularization, generalization, and uncertainty quantification in model predictions.

**Optimization Theory** - Helps understand gradient descent variants, convergence properties, and the optimization landscape of neural networks.

**Real Analysis** - Provides the theoretical foundation for the Universal Approximation Theorem and understanding function spaces.

**Information Theory** - Useful for understanding entropy-based loss functions and the information-theoretic perspective on learning.

While you don't need to master all these fields to use neural networks effectively, understanding these mathematical foundations will give you deeper insight into why and how neural networks work.
