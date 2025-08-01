# LLM Applications

English | [简体中文](./index_zh-CN.md)

Large language models have made great progress since the transformer was introduced, [explained here](../04/index.md). Since natural language is sequential information, it's perfect for transformer architecture-based neural network training. At its core, the model takes in a list of tokens (could be a word, punctuation, or a phrase, etc.) in their meaningful order, trying to predict the next token, which is the output from the calculation of the weights applied to the input tokens.

For example, if you give "I love " as input to a trained LLM model, it will try to predict the next token from its entire vocabulary (which typically contains tens of thousands of tokens). It might assign probabilities like 0.35 to "you", 0.15 to "sports", 0.08 to "music", 0.05 to "reading", and small probabilities to thousands of other possible tokens. The LLM application typically selects the token with the highest probability ("you" in this case) as the output. For the LLM application to continue, it takes "I love you" as the new input, predicts the next token (likely "." with high probability), resulting in the sentence "I love you.".

You may ask, if we use a loop like this to query the model, how does it know when it should stop? The trick is in the training data. We don't just feed the model pure text. We put in special tokens such as "\<|start|\>" "\<|end|\>" in the text to indicate the beginning and end of the text chunk, which count as tokens too. The model learns to output this type of token, then the LLM application breaks the loop when seeing this type of token. Thus, in LLM applications like ChatGPT, Claude, etc., you'll put in some query (we call it a prompt), and the model will output one word at a time. In the UI, it looks like a human trying to talk to you.

Now, let's get technical.

For the training process, the first step is called tokenization. For text inputs, we have to translate the information into something which models can understand, which are numbers, since the foundation of computers are math calculations. For a model, it maintains a list of tokens, mapping text to numbers. When a text chunk is provided, the tokenizer breaks it into token sequence then shows the list of the numbers for model to train from. When the model outputs, it's also outputting probabilities for a list of numbers, then we use the tokenizer to decode the number into token for display again.

For example, we have a token mapping list like this:

| Token | ID |
|-------|-----|
| <\|start\|> | 1 |
| I | 2 |
| love | 3 |
| you | 4 |
| sports | 5 |
| gold | 6 |
| . | 7 |
| <\|end\|> | 8 |

When training a model, we have an input "I love", then model actually sees `[1,2,3]`, then after calculation, it outputs probabilities for all possible next tokens. For example, it might output 0.5 for token 4 ("you"), 0.3 for token 5 ("sports"), and 0.2 for token 6 ("gold"), then we pick the highest probability answer 4, which is token "you".

Then the actual training is broken into the following steps:

**Pre-train**
In this step, the LLM learns the flow of text, which is natural language sequence. The model trainers obtain large amounts of text chunks from sources like internet, books, human writers, etc. Then they clean it and format it into structures like what we talked about above. For a chunk like "I love you.", a transformer-based model is trained using causal language modeling: the model sees the sequence up to a certain position and tries to predict the next token. For example, given "I love", the model should predict "you", and given "I love you", it should predict ".". The model calculates probabilities for all possible next tokens, and the training loss measures how well the predicted probability distribution matches the actual next token. Through backpropagation, the model adjusts its weights to minimize this loss. This process is repeated across massive amounts of text data until the model converges to acceptable performance. With billions of parameters, the model learns complex patterns and relationships between tokens, enabling it to generate coherent text. Since the model outputs probability distributions, it can produce varied responses to the same input.

**Supervised Fine Tuning**
After pre-training, the model has learned general language patterns but may not follow instructions well or produce helpful responses. In supervised fine-tuning (SFT), the model is trained on curated datasets of high-quality question-answer pairs, conversations, and instruction-following examples. This teaches the model to respond appropriately to user queries and follow specific formats. The training process is similar to pre-training but uses smaller, higher-quality datasets focused on desired behaviors.

For example, for chat models, the model needs to learn what is a user input, what should be its answer to that input. As the trainer, it will prepare lots of queries, then hire human labelers to create high quality answers for these queries. The conversation pair will be formatted into something like this:

```plain
<|im_start|>user<|im_sep|>What is a watermelon?<|im_end|><|im_start|>assistant<|im_sep|>It's a type of juicy fruit, green outside and red inside<|im_end|>
```

Then with this type of data, the model learns to distinguish questions and predict appropriate answers. With a proper UI, the LLM applications then will talk to you like a human. For pre-training stage, it could cost months and millions of dollars of computation power, however, for SFT, it typically takes days to weeks with proper data and significantly less computational resources. In summary, the trained model is a lossy imitation of the data provided by human labelers.

**Reinforcement Learning**
The final step often involves reinforcement learning from human feedback (RLHF). In general, it's a hard task for humans to create all the content, but it's much easier for human labelers to compare which output is better from a model. So for RLHF step, we ask the model to generate outputs for the same inputs for many iterations, then ask the human labelers to provide ranking for quality, helpfulness, and safety. These preferences provide rewards to train the language model using reinforcement learning algorithms like PPO (Proximal Policy Optimization, an algorithm that optimizes policies by taking multiple small update steps within a trusted region to avoid performance collapse while maximizing rewards). This process was generally used to help align the model's outputs with human values and preferences, making it more helpful, harmless, and honest.

This step has evolved to be used for training the model to be able to reason. In retrospect, the model only predicts one token at a time, so if we give a hard problem to the model and ask for the answer directly, we're asking the model to calculate an answer in one pass. However, if we ask the model to explain the details of why it arrives at the answer, then the model generates reasoning tokens step-by-step in sequence, which provides more context for subsequent token predictions and allows the model to leverage its learned patterns more effectively, making the answer more likely to be accurate.

The company DeepSeek has used this type of training in its reasoning model `deepseek-R1`. In its RLHF stage, the trainer provided data with explicit reasoning steps, then rewarded the model based on human preferences for both the reasoning process and final answers:

```plain
<|im_start|>user<|im_sep|>What is a watermelon?<|im_end|><|im_start|>assistant<|im_sep|><think>The user is asking about watermelons. I should provide a clear, factual description covering the key characteristics...</think>It's a large, round fruit with green skin and red flesh inside, commonly eaten in summer.<|im_end|>
```  

In their paper, the researchers found an "Aha moment" where the model found a point to re-generate a better answer since the whole thinking process provided more context for predicting the next token. This has drastically improved the reasoning performance of the model, thus resulting in more accurate answers.

A lot of the RLHF process is still in experiment. Ideally the preferences should be provided by humans all the time with massive generations from a model. This is industrially expensive, so some model trainers started to use AI feedback (AIF) - training separate AI models to evaluate outputs based on learned human preferences, then using these AI evaluators to provide feedback instead of human raters. However, the process is lossy so it introduces new problems like the reward models learn to cheat, then produce terrible answers. The whole industry is still trying to improve on RLHF, maybe in the future this step will be replaced by better solutions.
