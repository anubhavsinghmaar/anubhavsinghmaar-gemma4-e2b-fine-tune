# Concepts Explained — Everything Used in This Fine-Tuning Project

This file explains every major concept, technique, and tool used in this project. If you are new to LLM fine-tuning, reading this top to bottom will give you a solid foundation. Each section goes from the intuition first, then the technical detail.

---

## Table of Contents

1. What is Fine-Tuning?
2. Why Not Just Prompt the Base Model?
3. Full Fine-Tuning vs Parameter-Efficient Fine-Tuning (PEFT)
4. LoRA — Low-Rank Adaptation
5. QLoRA — Quantized LoRA
6. 4-bit Quantization Explained
7. The Alpaca Format
8. SFTTrainer and Supervised Fine-Tuning
9. Cross-Entropy Loss
10. Reading the Loss Curve
11. Gradient Accumulation
12. Learning Rate and Cosine Scheduler
13. Gradient Checkpointing
14. The Transformer Architecture (Brief)
15. Attention Modules and Why LoRA Targets Them
16. Gemma 4 E2B — The Base Model
17. Unsloth — Why It Makes This Faster
18. PEFT Library
19. The Adapter Pattern
20. What Happens at Inference

---

## 1. What is Fine-Tuning?

A large language model like Gemma 4 is pre-trained on hundreds of billions of tokens scraped from the internet. It learns grammar, facts, reasoning patterns, and a general understanding of the world. But it has no idea that your company exists, what your policies are, or how your teams communicate.

Fine-tuning is the process of continuing the training of a pre-trained model on a smaller, domain-specific dataset so the model adapts its behaviour to your use case. The model's existing knowledge is preserved — you are not starting from scratch. You are nudging it in a specific direction.

There are two main outcomes fine-tuning can produce:

- **Instruction following** — teaching the model to respond in a particular format or tone
- **Domain adaptation** — teaching the model to understand terminology, entities, and policies specific to a domain

This project aims for both: the model learns the Singhmaar Corp world and learns to respond in a concise, professional enterprise tone.

---

## 2. Why Not Just Prompt the Base Model?

You could provide all the company context in a system prompt every time you query the model. This is called in-context learning or retrieval-augmented generation (RAG). It works, but has real limitations:

- **Context window limits** — you can only fit so much text into a single prompt
- **Latency and cost** — long prompts are slower and more expensive at inference
- **Consistency** — the model's behaviour can vary based on how the prompt is phrased
- **No structural change** — the model does not actually learn your domain; it only borrows it temporarily from the prompt

Fine-tuning internalises the domain into the weights themselves. The model does not need to be reminded who Singhmaar Corp is every time — it already knows.

---

## 3. Full Fine-Tuning vs Parameter-Efficient Fine-Tuning (PEFT)

**Full fine-tuning** updates every single weight in the model. For a 4.36 billion parameter model like Gemma 4 E2B, this means:

- Storing all 4.36B parameters in GPU memory
- Storing gradients for all 4.36B parameters
- Storing the optimiser states (Adam stores two additional values per parameter)
- Total GPU memory requirement: often 60 to 100GB+ for a model this size

That is not feasible on a free T4 GPU with 15GB VRAM.

**Parameter-Efficient Fine-Tuning (PEFT)** is a family of techniques that update only a tiny fraction of the model's parameters while keeping the rest frozen. The key insight is that you do not need to change everything to meaningfully adapt the model — you just need to change the right things.

LoRA is currently the most widely used PEFT technique. This project uses QLoRA, which is LoRA combined with 4-bit quantisation.

---

## 4. LoRA — Low-Rank Adaptation

**Paper:** "LoRA: Low-Rank Adaptation of Large Language Models" — Edward Hu et al., 2021 (arXiv:2106.09685)

**The core insight:** When you fine-tune a neural network, the changes to the weight matrices tend to have a low "intrinsic rank." This means the weight update can be expressed as the product of two much smaller matrices rather than one large matrix.

**How it works technically:**

Instead of updating a large weight matrix W directly, LoRA freezes W and adds two smaller matrices A and B alongside it:

```
new output = W·x + (B·A)·x
```

Where:
- W is the original frozen weight matrix (e.g. 4096 × 4096)
- A is a small matrix of shape (4096 × r)
- B is a small matrix of shape (r × 4096)
- r is the rank — a small number like 8, 16, or 32

Only A and B are trained. W never changes.

**Why this is efficient:**

If W is 4096 × 4096, it has 16.7 million values. If r = 16, then A has 65,536 values and B has 65,536 values — a total of 131,072 values. That is 127x fewer parameters to train and store gradients for.

**In this project:**
- LoRA rank (r) = 16
- LoRA alpha = 16 (the scaling factor, alpha/r = 1.0)
- Total trainable parameters = 25.3 million
- Percentage of total model trained = 0.49%

**Key advantage LoRA has over earlier adapter methods:** The trained matrices A and B can be merged directly into W at inference time with no extra latency:

```
W_merged = W + B·A
```

After merging, the model runs at exactly the same speed as the original.

---

## 5. QLoRA — Quantized LoRA

**Paper:** "QLoRA: Efficient Finetuning of Quantized LLMs" — Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer, 2023 (arXiv:2305.14314)

QLoRA combines two ideas:

1. **Quantise the base model to 4-bit precision** — dramatically reducing the memory required to hold the frozen weights
2. **Train LoRA adapters at full (16-bit) precision** — keeping the trainable parts accurate

The result: you can fine-tune a 65 billion parameter model on a single 48GB GPU, or a 4 billion parameter model on a free 15GB T4 GPU — which is exactly what this project does.

**Three innovations QLoRA introduced:**

**NF4 (4-bit NormalFloat):** A new 4-bit data type designed specifically for normally distributed weights, which is how neural network weights tend to be distributed. Previous 4-bit formats wasted quantisation bins on extreme values. NF4 is information-theoretically optimal for this distribution.

**Double Quantisation:** Even the quantisation constants (the numbers used to rescale values back to full precision) are quantised. This saves an additional ~0.37 bits per parameter — about 3GB for a 65B model.

**Paged Optimisers:** Uses NVIDIA's unified memory to handle sudden memory spikes that occur during long sequence training, preventing out-of-memory crashes.

---

## 6. 4-bit Quantisation Explained

Quantisation is the process of representing a number using fewer bits.

A standard neural network weight is stored as a 32-bit float (FP32) or 16-bit float (FP16). A 4-bit integer can only represent 16 distinct values. The question is: how do you map the full range of a weight matrix (which might span from -2.5 to +2.5) into just 16 values without losing too much information?

The answer is normalisation: find the maximum absolute value in a block of weights, divide all values by that maximum, and then round to the nearest of your 16 quantised levels. During computation, you dequantise back to 16-bit for the actual calculation (the forward pass), but the weights sit in memory as 4-bit.

**Memory savings:**
- FP16: 2 bytes per weight × 4.36B weights = ~8.7 GB
- 4-bit: 0.5 bytes per weight × 4.36B weights = ~2.2 GB

This is what makes fitting the model into a T4 GPU possible.

---

## 7. The Alpaca Format

Alpaca is a dataset format for instruction fine-tuning developed by Stanford. It became a de facto standard because it is simple, human-readable, and supported by most fine-tuning frameworks.

Each example has three fields:

```json
{
  "instruction": "The task or question",
  "input": "Optional additional context",
  "output": "The ideal response"
}
```

During training, these are converted into a full conversation using the model's chat template. For Gemma 4, the conversion looks like this:

```
<start_of_turn>user
What is the maternity leave policy at A. Singhmaar Corp?
<end_of_turn>
<start_of_turn>model
Female employees are entitled to 26 weeks of paid maternity leave...
<end_of_turn>
```

The model is trained to predict only the assistant's response, not the user's message. This is called "response-only" training — the loss is computed only on the tokens the model is supposed to generate.

---

## 8. SFTTrainer and Supervised Fine-Tuning

**SFTTrainer** (Supervised Fine-Tuning Trainer) comes from Hugging Face's TRL library (Transformer Reinforcement Learning). Despite the name, SFTTrainer does not use reinforcement learning — it is a standard supervised learning loop adapted specifically for language model fine-tuning.

It handles:
- Converting datasets to the correct chat format
- Masking the user's tokens so loss is computed only on the assistant's response
- Mixed precision training (FP16 or BF16)
- Gradient accumulation
- Evaluation on a held-out split
- Checkpointing and best model selection

In this project, SFTTrainer is configured with:
- `eval_strategy='epoch'` — evaluate after each epoch
- `load_best_model_at_end=True` — keep the checkpoint with the lowest eval loss
- `metric_for_best_model='eval_loss'` — use validation loss as the selection criterion

---

## 9. Cross-Entropy Loss

During training, the model makes predictions one token at a time. At each step, it outputs a probability distribution over all possible next tokens (the vocabulary). The correct next token is known from the training data.

Cross-entropy loss measures how wrong the model's prediction is:

```
Loss = -log(probability assigned to the correct token)
```

If the model assigns 100% probability to the correct token, the loss is 0. If the model assigns near-zero probability to the correct token, the loss is very high.

Across a batch of sequences, all individual token losses are averaged to produce a single scalar loss value. The optimiser then adjusts the LoRA weights (A and B matrices) to reduce this loss.

**Why cross-entropy for language models?**
Language modelling is fundamentally a classification problem — classify the next token from a vocabulary of ~256,000 options. Cross-entropy is the standard loss function for multi-class classification.

---

## 10. Reading the Loss Curve

The loss curve from this training run:

| Epoch | Training Loss | Validation Loss |
|---|---|---|
| 1 | 13.43 | 5.73 |
| 2 | 6.42 | 5.07 |
| 3 | 4.64 | 4.93 |

**What these numbers mean:**

Training loss starts high (13.43) because the model's LoRA matrices are initialised randomly — it has no idea yet how to respond in the Singhmaar Corp style. By epoch 3 it has fallen to 4.64, a drop of about 65%.

Validation loss starts at 5.73 and ends at 4.93 — a 14% improvement. The gap between training loss (4.64) and validation loss (4.93) at epoch 3 is small, which is a good sign. A large gap would suggest overfitting — the model memorising the training examples rather than learning the underlying patterns.

**Why does training loss start higher than validation loss in epoch 1?**
Training loss is computed at the start of the epoch before any learning has happened on those batches. Validation loss is computed after the epoch, using the updated weights. By the time validation runs, the model has already improved.

**What would bad loss curves look like?**
- Validation loss going up while training loss goes down = overfitting
- Both losses barely moving = learning rate too low or dataset too small to have an effect
- Loss exploding to NaN = learning rate too high or numerical instability

---

## 11. Gradient Accumulation

Normally, you compute gradients from a batch of examples and update the weights immediately. The problem is that larger batches produce more stable gradient estimates, but fitting more examples into GPU memory at once has limits.

Gradient accumulation solves this by computing gradients from multiple small batches and adding them together before updating the weights. The effect is mathematically equivalent to using a larger batch.

In this project:
- Per-device batch size = 2
- Gradient accumulation steps = 4
- Effective batch size = 2 × 4 = 8

The weights are only updated every 4 steps. This means the training loop runs 4 forward passes, accumulates their gradients, then does 1 weight update — achieving the stability of a batch-8 training run on a GPU that can only hold 2 examples at once.

---

## 12. Learning Rate and Cosine Scheduler

**Learning rate (LR):** Controls how large each weight update step is. Too high and the model diverges — updates overshoot and the loss explodes. Too low and training is slow or stalls entirely.

This project uses LR = 2e-4 (0.0002), which is a standard starting point for LoRA fine-tuning.

**Cosine scheduler:** Rather than keeping the learning rate constant throughout training, a scheduler changes it over time. The cosine scheduler:

1. Starts at 0 and warms up to the target LR over the first few steps (warmup steps = 5 here)
2. Then follows a cosine curve, gradually decreasing to near 0 by the end of training

The intuition: early in training, the model is making large adjustments and a high LR is useful. Later in training, the model is fine-tuning small details and a lower LR prevents overshooting.

---

## 13. Gradient Checkpointing

Normally, during a forward pass, all intermediate activations are stored in GPU memory so they can be used during the backward pass (gradient computation). For a large model with long sequences, this can consume enormous amounts of memory.

Gradient checkpointing trades compute for memory: instead of storing all activations, only a subset (checkpoints) are stored. The rest are recomputed on-demand during the backward pass. This roughly halves memory usage at the cost of about 33% more computation.

Unsloth has its own optimised implementation of gradient checkpointing that is faster than the standard PyTorch version. This is one of the key reasons Unsloth achieves the speed it does on limited hardware.

---

## 14. The Transformer Architecture (Brief)

Every modern LLM, including Gemma 4, is built on the Transformer architecture (Vaswani et al., 2017). The key components relevant to fine-tuning are:

**Attention layers:** Allow the model to look at different positions in the input sequence when generating each output token. Each attention layer contains query (Q), key (K), value (V), and output (O) projection matrices. These are the primary targets for LoRA.

**MLP layers:** Feed-forward networks that apply non-linear transformations after attention. Contain gate, up, and down projection matrices. Also targeted by LoRA in this project.

**Embedding layers:** Convert token IDs to vectors and back. Usually not modified during LoRA fine-tuning.

**Layer normalisation:** Normalises activations between layers for training stability. Usually frozen.

---

## 15. Attention Modules and Why LoRA Targets Them

The original LoRA paper found that applying LoRA to the attention weight matrices (Q and V projections) was sufficient for strong performance on most tasks. Subsequent work showed that targeting all attention matrices (Q, K, V, O) plus the MLP layers gives better results for domain adaptation tasks.

In this project, Unsloth's `get_peft_model` is configured with:
- `finetune_attention_modules=True`
- `finetune_mlp_modules=True`
- `finetune_vision_layers=False` (Gemma 4 is multimodal but this is a text-only task)

Targeting more modules increases the number of trainable parameters but gives the model more capacity to adapt. With rank 16 and both attention and MLP modules, this project has 25.3M trainable parameters — enough for meaningful adaptation without risking overfitting on 100 examples.

---

## 16. Gemma 4 E2B — The Base Model

Gemma 4 is a family of open models released by Google DeepMind in April 2025. Key characteristics:

- **E2B** stands for approximately 2 billion effective parameters in a Mixture-of-Experts (MoE) architecture. The total parameter count is 4.36B but only a subset are active for any given input.
- **Multimodal:** Gemma 4 models can process both text and image inputs, though this project uses only the text modality.
- **Context window:** 256K tokens — significantly longer than most open models.
- **Multilingual:** Trained on over 140 languages.
- **License:** Apache 2.0, which allows commercial use, modification, and distribution.
- **Instruction-tuned variant:** The `-it` suffix means this is the chat-fine-tuned version, not the raw pretrained base. Starting from an instruction-tuned model means the model already knows how to follow instructions — we are only adding domain knowledge.

The Unsloth version (`unsloth/gemma-4-e2b-it`) is a memory-optimised version of the same model with Unsloth's custom kernels pre-integrated.

---

## 17. Unsloth — Why It Makes This Faster

Unsloth is an open-source library by Daniel and Michael Han that makes LLM fine-tuning significantly faster and more memory-efficient through custom CUDA kernels.

Key optimisations:
- **Custom attention kernel:** Rewrites the attention computation in pure CUDA, avoiding unnecessary memory copies that PyTorch's default implementation produces
- **Optimised gradient checkpointing:** Recomputes activations more efficiently than PyTorch's native implementation
- **Fused operations:** Combines multiple operations (like layer norm + attention) into single kernel calls, reducing memory bandwidth usage

**Results claimed by Unsloth (and widely validated by the community):**
- 2x faster training compared to standard HuggingFace + PEFT
- Up to 70% less VRAM usage

In this project, training 90 examples for 3 epochs took 3.4 minutes on a T4 GPU. The same configuration without Unsloth would take approximately 6 to 8 minutes and might not fit in T4 memory at all.

---

## 18. PEFT Library

PEFT (Parameter-Efficient Fine-Tuning) is a Hugging Face library that provides implementations of all major PEFT methods including LoRA, LoHa, LoKr, IA3, and prompt tuning.

In this project, PEFT is used indirectly through Unsloth's `FastModel.get_peft_model()`, which wraps PEFT's `LoraConfig` and `get_peft_model()` with Unsloth's optimisations.

After training, the adapter is saved as a `PeftModel` checkpoint, which consists of:
- `adapter_config.json` — the LoRA configuration (rank, alpha, target modules, etc.)
- `adapter_model.safetensors` — the actual trained A and B matrices (101MB in this project)

The base model is not saved — only the tiny adapter. Anyone who wants to run the fine-tuned model loads the base model separately and attaches the adapter on top.

---

## 19. The Adapter Pattern

The adapter pattern is a software design pattern where a small, swappable component modifies the behaviour of a larger, fixed component. LoRA adapters are a direct implementation of this in deep learning.

**What this means practically:**

```
Base model (4.36B params, ~8.7 GB)     [frozen, never changes]
      +
LoRA adapter (25M params, 101 MB)      [trained, domain-specific]
      =
Fine-tuned model (4.36B effective)     [behaves like Singhmaar Corp AI]
```

You can:
- Load the same base model and swap in different adapters for different tasks
- Share the 101MB adapter file without sharing the 8.7GB base model
- Merge the adapter into the base model for single-file deployment

This is why LoRA adapters on HuggingFace are so small compared to full model uploads — they are literally just the delta, not the whole model.

---

## 20. What Happens at Inference

When you load the fine-tuned model and ask it a question, here is what happens step by step:

1. **Tokenisation:** Your question is split into tokens (subword units). "What is the leave policy?" might become ["What", "is", "the", "leave", "policy", "?"].

2. **Embedding:** Each token ID is converted to a high-dimensional vector (the embedding).

3. **Forward pass through transformer layers:** The embeddings pass through each of Gemma's transformer layers. In each layer:
   - The attention mechanism looks at all previous tokens and decides which are most relevant
   - In the attention weight matrices, the LoRA adapter adds its contribution: `(W + B·A)·x`
   - The MLP layer applies a non-linear transformation
   
4. **Output logits:** After the final layer, the model produces a vector of logits — one per token in the vocabulary (~256K tokens). These are converted to probabilities via softmax.

5. **Token sampling:** One token is sampled from this probability distribution. With temperature=0.7, the distribution is slightly sharpened — common tokens are preferred but rare ones still have a chance.

6. **Repeat:** The sampled token is appended to the sequence and the model runs again for the next token. This continues until a stop token or the maximum length is reached.

7. **Decode:** The sequence of token IDs is converted back to readable text.

The LoRA adapter's influence is felt at every transformer layer where it is applied. The model's domain-specific knowledge is not stored in one place — it is distributed across all the trained weight matrices.

---

## Further Reading

- LoRA paper: arxiv.org/abs/2106.09685
- QLoRA paper: arxiv.org/abs/2305.14314
- Unsloth GitHub: github.com/unslothai/unsloth
- HuggingFace PEFT docs: huggingface.co/docs/peft
- TRL SFTTrainer docs: huggingface.co/docs/trl/sft_trainer
- Gemma 4 model card: huggingface.co/google/gemma-4-E2B-it

