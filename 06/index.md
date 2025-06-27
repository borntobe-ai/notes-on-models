# Reinforcement Learning

English | [简体中文](./index_zh-CN.md)

Reinforcement learning is a type of machine learning where an agent learns optimal strategies by interacting with an environment and receiving reward signals, with the goal of maximizing cumulative rewards. The key to reinforcement learning lies in the agent's need to balance between exploring unknown actions and exploiting known optimal actions. Here we use a specific example to explain this: the [MuZero](https://arxiv.org/abs/1911.08265) model proposed by DeepMind.

Before MuZero, there was already a Go-playing model trained through reinforcement learning called AlphaGo. It became famous after its match against world champion Lee Sedol, making a move that human players would never make. After analysis, this move proved to be a brilliant strategy, and this type of play had never appeared in Go history - it was AlphaGo's own prediction.

What makes MuZero special is that it doesn't use human gameplay data, but instead generates game data by itself. Based on corresponding environment feedback, it can learn to play a game like an expert without knowing the game rules. There's a [repository](https://github.com/werner-duvaud/muzero-general) on GitHub that provides the specific implementation of this model's training. Setting aside technical details, its implementation principle is as follows:

- First, implement a game environment that can tell users the current game state, who the player is, what actions can be executed next, how the game state updates if a certain action is executed, what the reward for the action is, and whether the game is over.
- Design the model architecture. The model has three internal components for predicting strategy, representing the environment, and selecting actions. After training, it learns the hidden states of the environment.
- Generate game data. Here the model is initialized with random parameters. The training code starts a random game, then lets the model predict different actions. For each action, environment feedback is provided, and rewards are fed back to the model. The model optimizes its parameters based on the reward results, and new games will use the new model parameters to predict actions. Through this repetition, the model learns the optimal strategy for playing the game.

The Farama Foundation provides an environment that helps with reinforcement learning, [Gymnasium](https://github.com/Farama-Foundation/Gymnasium). You don't need to build your own environment from scratch - you can directly reference it to debug your training. Of course, writing your own environment for specific tasks is also important.

Reinforcement learning is helpful in many fields. For example, in autonomous driving, we can't let cars crash around on roads to learn which driving behaviors are effective and which are harmful. So building virtual driving environments and letting algorithms learn to handle various signals through environment feedback is a recommended approach. Of course, models must also be tested after training to see if they overfit to the environment and cannot generalize outside of it.
