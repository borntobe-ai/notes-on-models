# Code first

English | [简体中文](./index_zh-CN.md)

A model is trained in the following steps in general:

1. **Define the architecture of the model**: Specify the type of neural network (e.g., transformer), the number of layers, the size of each layer, and other hyperparameters such as learning rate and batch size.
2. **Prepare and preprocess data**: Collect a large dataset and preprocess it by tokenizing text, removing noise, and converting it into a format suitable for the model.
3. **Initialize the model**: Set initial values for the weights and biases in the model.
4. **Forward pass**: Pass the input data through the model to generate predictions. The model processes the input data layer by layer to produce an output.
5. **Calculate the loss**: Compare the model's output with the desired output using a loss function (e.g., cross-entropy loss for classification tasks). The loss quantifies how far the model's predictions are from the actual targets.
6. **Backward pass (Backpropagation)**: Compute the gradients of the loss with respect to each of the model's parameters (weights and biases) by applying the chain rule of calculus. This step propagates the error backward through the network.
7. **Update weights**: Adjust the model's weights using an optimization algorithm (e.g., stochastic gradient descent, Adam) to minimize the loss. The optimizer updates the weights based on the computed gradients.
8. **Iterate over the dataset**: Repeat the forward pass, loss calculation, backward pass, and weight updates for many iterations (epochs) over the dataset. This involves feeding the model with batches of data and continually refining the weights.c
9. **Iterate over the dataset**: Repeat the forward pass, loss calculation, backward pass, and weight updates for many iterations (epochs) over the dataset. This involves feeding the model with batches of data and continually refining the weights.
10. **Model ready**: Once the training converges and the model performs well on both training and validation datasets, it is considered ready for deployment. Optionally, the model can undergo further fine-tuning on specific tasks or datasets.

Here is the data:

| Input x | Output y |
| ------- | -------- |
| 1       | 3        |
| 2       | 5        |
| 3       | 7        |

Find the function y=f(x) for input x and output y.

Assume the function generating y from x is y = ax + b. From the first two sets of data, we can calculate a = 2 and b = 1. Substituting these values into the third set of inputs and outputs confirms the correctness of a and b.

However, if new data x = 4, y = 8 were introduced, we see that y calculated from y = 2x + 1 is 9, which does not equal 8. This indicates that the assumed function is incorrect, and we need to adjust the calculation process to find the correct relationship between y and x.

Below is a simple code example to illustrate this process:

```python
import torch

x = torch.ones(5)  # input
y = torch.zeros(3)  # target output
w = torch.randn(5, 3, requires_grad=True)  # initial parameters
b = torch.randn(3, requires_grad=True)  # initial parameters

learning_rate = 0.01 # The learning rate controls the magnitude of changes to the model parameters (weights and biases) based on the estimated error during each update.
optimizer = torch.optim.SGD([w, b], lr=learning_rate)  # Stochastic Gradient Descent (SGD): Updates parameters using the gradient of the loss function with respect to the parameters.

steps = 0

while steps < 5:
    print(f"Training step {steps}: ")

    # Clear the gradients before the forward pass
    optimizer.zero_grad()

    z = torch.matmul(x, w) + b  # target function z = w * x + b
    loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)  # calculate loss
    loss.backward()  # backward pass

    print(w.grad)
    print(b.grad)

    # update weights
    optimizer.step()

    steps += 1
```

Further reading:

- [llm.c](https://github.com/karpathy/llm.c) is a simple implementation of a transformer model for language modeling. Karpathy has detailed explanations of the code and the math behind it.

- [PyTorch Tutorial](https://pytorch.org/tutorials/beginner/basics/intro.html)
