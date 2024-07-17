English | [简体中文](./index_zh-CN.md)

# Always code first

[Reference Code](https://github.com/karpathy/llm.c)

A LLM model is trained in the following steps in general:

1. **Define the architecture of the model**: Specify the type of neural network (e.g., transformer), the number of layers, the size of each layer, and other hyperparameters such as learning rate and batch size.
2. **Prepare and preprocess data**: Collect a large dataset and preprocess it by tokenizing text, removing noise, and converting it into a format suitable for the model. This step doesn't necessarily involve labeling data explicitly as input-output pairs, as LLMs typically use unsupervised or self-supervised learning.
3. **Initialize the model**: Set initial values for the weights and biases in the model, usually using a method like Xavier initialization or He initialization.
4. **Forward pass**: Pass the input data through the model to generate predictions. The model processes the input data layer by layer to produce an output.
5. **Calculate the loss**: Compare the model's output with the desired output using a loss function (e.g., cross-entropy loss for classification tasks). The loss quantifies how far the model's predictions are from the actual targets.
6. **Backward pass (Backpropagation)**: Compute the gradients of the loss with respect to each of the model's parameters (weights and biases) by applying the chain rule of calculus. This step propagates the error backward through the network.
7. **Update weights**: Adjust the model's weights using an optimization algorithm (e.g., stochastic gradient descent, Adam) to minimize the loss. The optimizer updates the weights based on the computed gradients.
8. **Iterate over the dataset**: Repeat the forward pass, loss calculation, backward pass, and weight updates for many iterations (epochs) over the dataset. This involves feeding the model with batches of data and continually refining the weights.c
9. **Iterate over the dataset**: Repeat the forward pass, loss calculation, backward pass, and weight updates for many iterations (epochs) over the dataset. This involves feeding the model with batches of data and continually refining the weights.
10. **Model ready**: Once the training converges and the model performs well on both training and validation datasets, it is considered ready for deployment. Optionally, the model can undergo further fine-tuning on specific tasks or datasets.
