# Build Your Own Application

English | [简体中文](./index_zh-CN.md)

Let's build a simple search agent that looks up current information to answer your questions.

[Explore the demo](https://github.com/borntobe-ai/simple-agent)

## The Basic Loop

The core interaction is straightforward: you send a prompt to a model, and it can answer using its existing knowledge and the conversation context. If you also give it access to tools, it can request a tool call to gather information or perform an action.

Your application executes the requested tool and sends the result back to the model. The model can then request another tool call or produce a final answer. This cycle is the foundation of a simple agent.

## Four Message Roles

To support this loop, we organize the conversation into four message roles:

- **System**: instructions that define the agent's behavior.
- **User**: questions or requests from the user.
- **Assistant**: responses from the model, including requests to call tools.
- **Tool**: results returned by the tools the application executes.

Together, these messages form the conversation history, giving the model context for each step.

## Connecting the Pieces

An **orchestrator** coordinates the model, tools, and conversation history. For each new user message, it:

1. Adds the message to the conversation history.
2. Sends the history and available tool definitions to the AI provider.
3. Appends the model's response to the history.
4. If the model requests tool calls, executes them, adds their results to the history, and asks the model again.
5. Ends the turn when the model returns a final text response, or when an error or a step limit stops the loop.

In the demo, we use DeepSeek Flash as the model and DeepSeek's web search tool to retrieve current information. The orchestrator connects them into a simple search agent.

At its simplest, an agent is software that uses a model and tools to carry out a task based on your prompt. The model decides which tools to request, and the application handles their execution and feeds the results back.

Get a [DeepSeek API key](https://platform.deepseek.com/), follow the setup instructions in the demo, and try a question that needs current information. You'll see the model–tool loop in action.
