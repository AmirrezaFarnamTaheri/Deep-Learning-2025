# Deep Learning Final Project

This repository section contains the complete, desloped, and unified implementations for the **Neural Networks and Deep Learning Final Project** (Tehran Institute for Advanced Studies - TeIAS, Instructor: Dr. Behnam Bahrak).

---

## Project Structure

```text
Final-Project/
├── 01_Instruction_Fine_Tuning.ipynb   # Problem 1: LLM Instruction Tuning on Persian SlimOrca
├── 02_Image_Captioning.ipynb          # Problem 2: Vision-Language Image Captioning (CNN-LSTM, Attention, Transformer)
├── runs_q1/                           # Training logs, checkpoints, and split index caches
└── README.md                          # Project documentation and empirical summary
```

---

## Problem 1: Instruction Fine-Tuning in Large Language Models

### Theoretical Framework & Objectives
The objective is parameter-efficient fine-tuning (PEFT) of a generative language foundation model (`google/gemma-2-2b-it`) on the conversational **SlimOrca Persian 50k** dataset, ensuring the model aligns with Persian instructional requests.

### Core Architectural Components
1. **Assistant-Only Loss Masking:**
   - Evaluates cross-entropy loss exclusively on target assistant tokens.
   - User prompt tokens and system instructions are masked with `labels = -100`, eliminating gradient noise from memorizing prompts.
2. **Three Fine-Tuning Regimes:**
   - **Soft Prompts (Prompt Tuning):** Freezes all 2B parameters and optimizes $p=20$ virtual continuous prompt tokens prepended to input embeddings ($40,960$ trainable parameters, $\approx 0.0016\%$).
   - **Low-Rank Adaptation (LoRA):** Injects trainable low-rank decomposition rank decomposition matrices into attention and MLP projections ($W = W_0 + \frac{\alpha}{r} B A$, $r=8$, $\alpha=16$, $9,830,400$ trainable parameters, $\approx 0.37\%$).
   - **Selective Layer Fine-Tuning:** Freezes the embedding layer and lower transformer blocks while unfreezing the top 2 transformer layers and the language modeling head ($\approx 8.5\%$ trainable parameters).
3. **Evaluation Protocol:**
   - Qualitative evaluation on standardized Persian benchmark prompts (reasoning, scientific explanations, code generation).
   - Quantitative evaluation measuring validation perplexity, Distinct-1/Distinct-2 n-gram ratios, and lexical repetition rates.

---

## Problem 2: Neural Image Captioning

### Theoretical Framework & Objectives
Image captioning bridges computer vision and natural language generation. This problem implements and compares three generations of captioning architectures on the **Flickr8k** benchmark.

### Models Implemented
1. **Architecture 1: CNN-LSTM Baseline (Show and Tell)**
   - **Encoder:** Pretrained ResNet-50 producing a global 2048-dimensional feature representation via Global Average Pooling.
   - **Decoder:** Word embedding layer followed by an LSTM cell with teacher forcing.
2. **Architecture 2: CNN-LSTM with Spatial Attention (Show, Attend and Tell)**
   - **Encoder:** Retains spatial $7 \times 7 \times 2048$ feature maps ($L=49$ spatial locations).
   - **Attention Mechanism:** Additive (Bahdanau) attention dynamically producing spatial weights $\alpha_{t, i} = \text{softmax}(v_a^\top \tanh(W_a h_{t-1} + U_a v_i))$.
   - **Interpretability:** Visualizes attention heatmaps over the original image for each generated word.
3. **Architecture 3: Vision-Transformer Captioner**
   - **Encoder:** 2D spatial feature projection to $d_{\text{model}} = 512$ with 2D positional encodings.
   - **Decoder:** Multi-head self-attention with causal look-ahead mask, multi-head cross-attention attending to spatial image memory, and feed-forward networks.

### Empirical Benchmarks
| Architecture | Visual Feature | Sequence Paradigm | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | Interpretability |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **CNN-LSTM Baseline** | Global Pool (2048-d) | Recurrent Bottleneck | 58.4 | 37.1 | 23.2 | 14.5 | Low |
| **CNN-LSTM + Spatial Attention** | Spatial Grid ($7 \times 7$) | Dynamic Alignment | 66.8 | 46.5 | 31.9 | 21.4 | High (Heatmaps) |
| **CNN-Transformer** | Spatial Patches ($49 \times 512$) | Multi-Head Self/Cross-Attn | **69.2** | **49.7** | **34.8** | **24.1** | Moderate |

---

## Requirements & Environment Setup

```bash
# Core deep learning libraries
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets peft accelerate bitsandbytes
pip install numpy pandas matplotlib seaborn tqdm pillow nltk
```
