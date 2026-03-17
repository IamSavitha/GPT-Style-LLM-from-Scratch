# GPT-Style Language Model from Scratch 

This project implements a **decoder-only GPT-style Transformer** from scratch using PyTorch. The model is trained on the **TinyStories dataset** to generate coherent short stories using character-level tokenization.

---

## Project Overview

This project explores how **autoregressive language models** work at a fundamental level by building a GPT-style architecture **without using any pre-trained models or embeddings**.

The model learns to predict the next character in a sequence and generates text by sampling from learned probability distributions.

---

## Objectives

- Build a **GPT-style Transformer** from scratch  
- Understand **self-attention and sequence modeling**
- Train on a real dataset (**TinyStories**)  
- Generate **coherent short stories**
- Analyze **failure modes in generation**

---

## Model Architecture

The model follows a **decoder-only Transformer architecture** consisting of:

- Token + Positional Embeddings  
- 6 Transformer Blocks  
- Multi-Head Self-Attention (8 heads)  
- Feed-Forward Neural Network (GELU activation)  
- Layer Normalization + Residual Connections  
- Language Modeling Head  

### Key Parameters

| Parameter | Value |
|----------|------|
| Embedding Dimension | 256 |
| Attention Heads | 8 |
| Transformer Layers | 6 |
| Sequence Length | 128 |
| FFN Dimension | 1024 |
| Dropout | 0.1 |
| Total Parameters | ~4.8M |

---

## Dataset

- **TinyStories Dataset**
- ~2 million short synthetic stories

### Split

- Train: 100,000 stories  
- Validation: 10,000 stories  

### Tokenization

- Character-level tokenization  
- Vocabulary size: **110 characters**

---

## Training Setup

- Optimizer: **AdamW**
- Learning Rate: `3e-3`
- Weight Decay: `1e-2`
- Batch Size: `128`
- Epochs: `20`
- Gradient Clipping: `1.0`

### Learning Rate Schedule

- Linear warmup (first 200 steps)
- Cosine decay to minimum `3e-4`

---

## Results

| Metric | Value |
|--------|------|
| Final Training Loss | 0.6057 |
| Final Validation Loss | 0.6438 |

- Stable convergence across 20 epochs  
- Minimal overfitting (small train-val gap)

---

## Sample Generations

### Low Temperature (T = 0)
- Fluent and coherent stories  
- Deterministic outputs  

### Medium Temperature (T = 0.8)
- More creative and diverse outputs  
- Mostly coherent  

### High Temperature (T = 1.2)
- Increased randomness  
- Occasional incoherence and nonsense  

---

## Failure Analysis

### 1. Repetition
She tried to pull it out, but it was too hard...
(repeats endlessly)

### 2. Loss of Coherence
- Sudden topic or character shifts  

### 3. Hallucination
- Non-existent words (e.g., *“ductors”, “wugged”*)

---

## Strengths

- Stable training without divergence  
- Generates grammatically correct sentences  
- Efficient architecture with ~4.8M parameters  
- Good generalization  

---

## Limitations

- Character-level tokenization limits semantic understanding  
- Context window limited to 128 tokens  
- Domain-specific (story generation only)  

---

## Tech Stack

- Python  
- PyTorch  
- NumPy  
- Matplotlib  

---

##  Project Structure

GPT-LLM/
│
├── data/
├── checkpoints
├── code_file.ipynb
├── model.pth
├──tinygpt_checkpoint.pt
└── README.md


---

##  How to Run

### 1. Install Dependencies
pip install torch numpy matplotlib

### 2. Train Model
python train.py

### 3. Generate Text
python generate.py

---

## References

- Vaswani et al., *Attention is All You Need* (2017)  
- TinyStories Dataset (2023)  
- GPT Architecture (OpenAI)

---

## Key Learnings

- Understanding **self-attention from first principles**
- Importance of **tokenization choices**
- Trade-offs between **coherence vs creativity (temperature)**
- Handling **training stability in transformers**

---

## Future Improvements

- Switch to **subword tokenization (BPE)**
- Increase **context window**
- Add **attention visualization**
- Train on **larger datasets**

---

## Author

**Keith Rajesh Gonslaves**

**Savitha Vijayarangan**  
