# Deep Learning 2025

This repository contains my personal answers, implementations, and submissions for the **Neural Networks and Deep Learning** course offered by the Department of Data Science at the Tehran Institute for Advanced Studies (TeIAS), instructed by Dr. Behnam Bahrak.

---

## Repository Structure

```text
Deep-Learning-2025/
├── Assignment 1/       # Multi-Layer Perceptrons & Manual Backpropagation
├── Assignment 2/       # Convolutional Neural Networks (CNNs), ISIC Skin Cancer Detection
├── Assignment 3/       # Recurrent Neural Networks (RNN, LSTM, GRU) for Time Series
├── Assignment 4/       # NLP with Attention Mechanisms and Transformers (BERT)
├── Assignment 5/       # Autoencoders, Generative Adversarial Networks (GANs), Diffusion Models
├── Assignment 6/       # Deep Reinforcement Learning (DQN, Dueling Munchausen DQN for Robotics)
└── Final-Project/      # Final Project (Desloped & Cleaned):
    ├── 01_Instruction_Fine_Tuning.ipynb  # Part 1: Instruction Fine-Tuning (Gemma-2-2B on Persian SlimOrca)
    ├── 02_Image_Captioning.ipynb         # Part 2: Neural Image Captioning (CNN-LSTM, Attention, Transformer)
    ├── runs_q1/                          # Checkpoints and evaluation cache
    └── README.md                         # Final project technical report
```

---

## Coursework Breakdown

### Assignment 1: Neural Network Foundations
- Multi-Layer Perceptrons (MLP) architecture design from scratch using NumPy.
- Explicit matrix calculus derivations for forward and backward passes.
- Mini-batch stochastic gradient descent (SGD) and learning rate scheduling.

### Assignment 2: Computer Vision with CNNs
- Custom CNN architecture implementation in PyTorch.
- Binary and fine-grained image classification on Cats vs Dogs and the ISIC Skin Cancer dataset.
- Data augmentation, dropout, batch normalization, and transfer learning.

### Assignment 3: Sequence Modeling & Recurrent Networks
- Vanilla RNN, Long Short-Term Memory (LSTM), and Gated Recurrent Unit (GRU).
- Time-series prediction and sequential pattern recognition.
- Vanishing/exploding gradient mitigation and gradient clipping.

### Assignment 4: Attention Mechanisms & Transformers
- Scaled dot-product attention and multi-head self-attention mechanisms.
- Bidirectional Encoder Representations from Transformers (BERT) fine-tuning.
- Contextual word embeddings and token classification.

### Assignment 5: Generative Modeling & Latent Representations
- Variational Autoencoders (VAEs) and latent space interpolation.
- Generative Adversarial Networks (GANs) with Wasserstein distance (WGAN-GP).
- Principles of Denoising Diffusion Probabilistic Models (DDPM).

### Assignment 6: Deep Reinforcement Learning
- Deep Q-Networks (DQN) with experience replay and target networks.
- Dueling DQN and Munchausen DQN variants applied to robotic motion control.

---

## Final Project
Located in [`Final-Project/`](Final-Project/):
1. **Instruction Fine-Tuning of LLMs:**
   - Pretrained model: `google/gemma-2-2b-it`.
   - Dataset: `miladmim/slim-orca-dedup-chat-50k-persian`.
   - Methods: Assistant-only loss masking, Soft Prompts (Prompt Tuning), Low-Rank Adaptation (LoRA), and Selective Layer Fine-Tuning.
2. **Vision-Language Image Captioning:**
   - Dataset: Flickr8k benchmark.
   - Three generations of architectures: CNN-LSTM Baseline, Spatial Additive Attention (Show, Attend and Tell), and Vision-Transformer Captioner.
   - Evaluated using Corpus BLEU (BLEU-1 to BLEU-4) and visual attention heatmap overlays.

---

## License
This project is licensed under the MIT License.
