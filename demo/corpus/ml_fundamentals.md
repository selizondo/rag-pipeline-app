# Machine Learning Fundamentals

## Supervised Learning

Supervised learning is a machine learning paradigm where a model is trained on labeled data — input-output pairs. The goal is to learn a mapping from inputs X to outputs Y such that the model generalizes to unseen inputs. Common examples include image classification, spam detection, and price prediction.

Key components: training set, validation set, test set. The model optimizes a loss function (e.g., cross-entropy for classification, MSE for regression) using gradient descent.

## Unsupervised Learning

Unsupervised learning finds structure in unlabeled data. Common approaches include clustering (K-Means, DBSCAN), dimensionality reduction (PCA, t-SNE, UMAP), and generative modeling (VAEs, GANs). The model has no ground-truth labels; it discovers patterns through similarity, density, or reconstruction.

## Overfitting and Regularization

Overfitting occurs when a model learns the training data too well — including noise — and fails to generalize to new data. Signs: high training accuracy, low validation accuracy (large gap).

Mitigations:
- **L1 (Lasso):** adds |w| penalty; drives sparse weights; performs feature selection
- **L2 (Ridge):** adds w² penalty; shrinks weights uniformly; more stable gradient
- **Dropout:** randomly zeros activations during training; forces redundant representations
- **Early stopping:** halt training when validation loss stops improving
- **Data augmentation:** increases effective dataset size

## Bias-Variance Tradeoff

The expected prediction error decomposes as:
  Error = Bias² + Variance + Irreducible Noise

- **High bias (underfitting):** model too simple, misses signal. Fix: add capacity, better features.
- **High variance (overfitting):** model too complex, memorizes noise. Fix: regularize, add data.
- **Goal:** find the complexity that minimizes total error on unseen data.

## Gradient Descent

Gradient descent is the backbone optimization algorithm for training neural networks. At each step, compute the gradient of the loss w.r.t. model parameters and move in the negative gradient direction.

Variants:
- **Batch GD:** gradient over full dataset; stable but slow
- **Stochastic GD (SGD):** gradient on single example; noisy but fast
- **Mini-batch GD:** gradient over a batch (32–256 examples); best practical tradeoff

Common optimizers: Adam (adaptive learning rates per parameter, momentum), AdaGrad, RMSProp. Adam is the default for most deep learning tasks.

## Cross-Validation

Cross-validation estimates how well a model generalizes to an independent dataset. K-fold CV splits data into K folds, trains K times (each time one fold is held out as validation), and averages the scores.

Stratified K-fold ensures class proportions are preserved in each fold — important for imbalanced datasets. Leave-one-out CV (LOOCV) is K-fold with K=n; unbiased but computationally expensive.

---

# Deep Learning

## Neural Networks

A neural network is a stack of parameterized transformations: Linear → Activation → Linear → Activation → ... → Output. Each layer computes h = activation(Wx + b). Non-linear activations (ReLU, GELU, sigmoid) allow networks to approximate arbitrary functions.

Backpropagation computes gradients via the chain rule, propagating error backwards from output to input layers. Modern frameworks (PyTorch, JAX) handle this automatically via autograd.

## Attention Mechanism

The attention mechanism computes a weighted sum of values, where weights depend on the similarity between a query and a set of keys:

  Attention(Q, K, V) = softmax(QK^T / √d_k) V

- Q (query): what we're looking for
- K (key): what each token "advertises"
- V (value): the actual content to aggregate

Self-attention: Q, K, V all come from the same sequence. Allows each token to attend to every other token, capturing long-range dependencies. Multi-head attention runs H independent attention heads in parallel, concatenates outputs, and projects back to model dimension.

## Transformer Architecture

Transformers (Vaswani et al., 2017) replaced RNNs as the dominant architecture for sequence modeling. Key components:

1. **Token + positional embeddings** — encode token identity and position
2. **Multi-head self-attention** — all-to-all token interaction; O(n²) cost
3. **Feed-forward sublayer** — per-token MLP (typically 4× hidden dim)
4. **Layer normalization + residual connections** — stabilize training
5. **Causal masking** (decoder) — prevents attending to future tokens

Pre-norm vs post-norm: modern models (GPT-3, LLaMA) use pre-norm (LayerNorm before attention/FFN) for training stability.

## Batch Normalization and Layer Normalization

**Batch norm (BN):** normalizes across the batch dimension for each feature. Reduces internal covariate shift, enables higher learning rates. Sensitive to batch size; behaves differently at train vs test time.

**Layer norm (LN):** normalizes across the feature dimension for each example. Independent of batch size; consistent behavior at train and test. Preferred for NLP/transformers where batch size can be 1 and sequence lengths vary.

## Vanishing and Exploding Gradients

In deep networks, gradients can shrink (vanish) or grow (explode) exponentially as they propagate back through layers. This makes training unstable or impossible.

Solutions:
- **Residual connections (skip connections):** add the input directly to the layer output, creating a gradient highway
- **Gradient clipping:** cap gradient norm before the optimizer step
- **Careful initialization (Xavier, He):** set initial weights to preserve activation variance
- **LSTM gating:** in recurrent nets, gates control gradient flow

---

# Retrieval-Augmented Generation (RAG)

## What Is RAG?

RAG combines a retrieval system with a generative language model. Instead of relying solely on the model's parametric (baked-in) knowledge, RAG retrieves relevant documents at inference time and includes them in the prompt as context.

Pipeline:
1. **Ingest:** chunk documents → embed → store in vector DB
2. **Retrieve:** embed query → find nearest chunks by cosine similarity
3. **Generate:** pass retrieved context + query to LLM → stream answer

Benefits: keeps model knowledge up-to-date without retraining; reduces hallucination by grounding answers in retrieved evidence; traceable — you can cite sources.

## RAG vs Fine-Tuning

| Aspect | RAG | Fine-Tuning |
|--------|-----|-------------|
| Knowledge update | Add docs to index; no retraining | Requires new training run |
| Hallucination risk | Lower (grounded in retrieved context) | Higher (memorized facts can be wrong) |
| Cost | Storage + retrieval at inference | Training compute + serving cost |
| Best for | Dynamic knowledge, factual Q&A, citations | Specialized style/format, domain-specific behavior |

In practice: combine both. Fine-tune for style and task format; use RAG for knowledge.

## Chunking Strategies

Documents must be split into chunks before embedding. Chunking strategy significantly impacts retrieval quality.

- **Fixed word-size chunks (256–512 words):** simple, consistent; can split mid-sentence
- **Sentence-based:** preserves sentence boundaries; chunks vary in size
- **Paragraph-based:** preserves topic coherence; chunks can be too long for embedding models
- **Sliding window with overlap:** overlapping chunks improve recall at boundaries

Key finding (from chunking experiment): chunk size 256 + overlap 32 outperforms 512 + overlap 64 on eval metrics. Smaller chunks have more specific embeddings; larger chunks dilute the embedding signal and reduce retrieval precision.

---

# LLM Fine-Tuning

## LoRA (Low-Rank Adaptation)

LoRA is a parameter-efficient fine-tuning method. Instead of updating all model weights W, LoRA freezes W and learns a low-rank decomposition of the weight update:

  W' = W + ΔW = W + B·A,  where A ∈ ℝ^(r×d), B ∈ ℝ^(d×r), r ≪ d

- Only A and B are trained (2×r×d parameters, << d² for full fine-tuning)
- Rank r controls the expressiveness/parameter tradeoff. Typical: r=8 to r=64
- Applied to attention weight matrices (Q, K, V, output projection)
- At inference: merge W + BA for zero added latency

LoRA enables fine-tuning 7B+ parameter models on a single consumer GPU that couldn't fit the full model in FP32.

## QLoRA

QLoRA (Dettmers et al., 2023) combines LoRA with 4-bit quantization (NF4 format) and double quantization of the quantization constants. The base model is loaded in 4-bit; LoRA adapters train in bfloat16.

This reduced Llama-2 7B fine-tuning memory from ~28 GB (full FP16) to ~6 GB (QLoRA) — enabling fine-tuning on a single T4 GPU (16 GB VRAM, available free on Google Colab).

Tradeoffs: 4-bit quantization introduces small accuracy degradation. QLoRA-fine-tuned models typically perform within 1–2 points of full fine-tuned models on standard benchmarks.
