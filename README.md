# Part 3: NLP and Sequence Modeling Mini Project

**Dataset:** SMS Spam Collection (5,574 records | Binary Classification: ham / spam)  
**Task:** Build an end-to-end NLP pipeline — from raw text to sequence models — and compare approaches.

---

## Project Structure

```
part-3-nlp-sequence-modeling/
├── README.md
├── notebook.ipynb          ← Full pipeline with explanations
├── requirements.txt
└── results/
    ├── class_distribution.png
    ├── text_length_distribution.png
    ├── model_evaluation.png       ← Bar chart comparing all models
    ├── model_evaluation.csv       ← Accuracy / F1 / Precision / Recall
    ├── confusion_matrices.png
    ├── lstm_training_history.png
    └── sample_predictions.txt
```

---

## Tasks Completed

### Task 1 – Dataset Understanding
- **5,574 records** with two columns: `label` (ham/spam) and `text`
- **Classes:** `ham` (4,827 — 86.6 %) · `spam` (747 — 13.4 %)
- **Average message length:** ~59 characters / ~6 tokens after cleaning

### Task 2 – Text Preprocessing
Pipeline applied to every message:
1. **Lowercasing** — normalise case
2. **URL removal** — strip `http://…` / `www.…` patterns
3. **Symbol stripping** — keep only `[a-z\s]`
4. **Whitespace normalisation**
5. **Tokenisation** — split on whitespace
6. **Stopword removal** — English function words removed
7. **Short-token removal** — tokens with ≤ 1 character discarded

### Task 3 – Text Vectorization

| Method | Description |
|--------|-------------|
| **TF-IDF** (uni + bigrams, 5 000 features) | Weights rare-but-important terms; used by LR |
| **Bag of Words** (5 000 features) | Raw term counts; used by Naive Bayes |
| **Tokenizer + Padding** (10 000 vocab, len 100) | Integer sequences for LSTM |

**Why must text be vectorized?**  
ML/DL models operate on numeric tensors. Raw strings carry no mathematical meaning. Vectorization maps words to numbers while preserving (some) semantic and statistical signal — TF-IDF captures term importance; embeddings capture semantic proximity; sequences preserve word order.

### Task 4 – Baseline Models

| Model | Accuracy | F1 (spam) | Precision | Recall |
|-------|----------|-----------|-----------|--------|
| Logistic Regression + TF-IDF | **1.000** | 1.000 | 1.000 | 1.000 |
| Naive Bayes + BoW            | **1.000** | 1.000 | 1.000 | 1.000 |

Both classical models achieve perfect scores on this dataset, confirming that spam messages contain distinctive vocabulary that TF-IDF / BoW capture extremely well.

### Task 5 – LSTM Sequence Model

#### Architecture

```
Input  →  [batch, 100]          integer token sequences
   ↓
Embedding(10 000, 64)           learnable dense word vectors
   ↓
SpatialDropout1D(0.3)           regularisation over feature maps
   ↓
LSTM(64, dropout=0.2, recurrent_dropout=0.2)
   ↓
Dense(32, activation='relu')
   ↓
Dropout(0.3)
   ↓
Dense(1, activation='sigmoid')  spam probability ∈ [0, 1]
```

| Component | Role |
|-----------|------|
| Embedding layer | Maps each integer token to a 64-dim vector; trained end-to-end |
| LSTM layer | Processes the sequence left-to-right, retaining hidden state across all 100 time steps |
| Dense + Dropout | Non-linear classification head with regularisation |
| **Loss function** | Binary cross-entropy |
| **Metric** | Accuracy + F1 Score |

**Training:** EarlyStopping (patience = 3), batch size 64, up to 10 epochs.  
The LSTM converges to ~87% accuracy — lower than the classical baselines because the dataset is small and its vocabulary patterns are strong enough for bag-of-words to dominate.

### Task 6 – Attention & Transformer Reflection

#### Why RNNs struggle with long-term dependencies
RNNs pass information through a single hidden state vector. As sequences grow, gradients must flow back through many time steps; they either **vanish** (become ≈ 0, so early tokens are forgotten) or **explode** (become very large, destabilising training). This is the *vanishing/exploding gradient problem*.

#### How LSTMs help
LSTMs introduce a **cell state** — a separate memory highway — and three learnable gates:
- **Forget gate** — decides what to erase from memory
- **Input gate** — decides what new information to write
- **Output gate** — decides what to expose to the next layer

The cell state can carry information across hundreds of steps with minimal multiplicative decay, substantially alleviating the vanishing gradient issue.

#### What attention solves in seq-to-seq tasks
In encoder–decoder models (e.g. translation), the encoder compresses the entire source into one fixed-size vector — a bottleneck that hurts long sequences. **Attention** lets the decoder *directly query* every encoder hidden state and compute a weighted sum, so no information is lost. Each output token attends to the most relevant source positions, enabling far better handling of long-range dependencies.

#### Why transformers matter in modern NLP and GenAI
Transformers replace recurrence entirely with **multi-head self-attention**, which computes pairwise relationships between all tokens in parallel — O(1) sequential steps vs O(n) for RNNs. This gives:

- **Parallelism** — orders-of-magnitude faster training on GPUs/TPUs
- **Global context** — every token attends to every other token from the first layer
- **Scalability** — the architecture scales to billions of parameters (GPT-4, Claude, Gemini, LLaMA, etc.)
- **Transfer learning** — pre-train once on vast corpora, fine-tune on any downstream task

Transformers are the foundation of all modern large language models (LLMs), enabling few-shot and zero-shot generalisation, instruction following, code generation, summarisation, and multimodal reasoning.

---

## Results Summary

See `results/model_evaluation.csv` and `results/model_evaluation.png` for the full comparison.

Key finding: classical TF-IDF + Logistic Regression achieves perfect classification on this dataset because spam vocabulary is highly distinctive. The LSTM reaches ~87% accuracy — lower because the small dataset and repetitive patterns favour memorisation of keywords over sequence learning. On larger, noisier datasets (e.g. sentiment analysis of movie reviews), sequence models and transformers substantially outperform bag-of-words baselines.

---

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```
