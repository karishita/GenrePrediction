# Explainable Multi-Label Genre Prediction using RoBERTa and Neural Topic Modeling

## Overview

This project performs **multi-label genre prediction** from book and movie summaries while providing **topic-based explanations** for its predictions.

Unlike traditional genre classifiers that only output labels, this model learns a latent topic space and explains predictions through interpretable topics represented by meaningful words extracted from the training corpus.

The deployed application is built using **Streamlit** and allows users to:

* Enter a book or movie summary
* Predict multiple genres simultaneously
* View genre probabilities
* Explore dominant latent topics
* Understand why a genre was predicted through topic-word explanations

---

## Architecture

### 1. Input Summary

The model takes a textual summary as input.

Example:

> A young wizard discovers a hidden kingdom and embarks on a dangerous quest to defeat an ancient dragon.

---

### 2. RoBERTa Contextual Encoding

The summary is tokenized using the RoBERTa tokenizer and passed through a pretrained RoBERTa encoder.

For each token, RoBERTa generates a contextual embedding.

```text
Input Summary
      ↓
RoBERTa Tokenizer
      ↓
RoBERTa Encoder
      ↓
Token Embeddings
```

---

### 3. Mean Pooling

 the model computes a single document representation by averaging the contextual token embeddings.

```text
Token Embeddings
      ↓
Mean Pooling
      ↓
Document Embedding
```

This document embedding captures the overall semantic meaning of the summary.

---

### 4. Variational Topic Modeling

The pooled embedding is projected into a latent topic space.

A neural encoder predicts:

* μ (mean)
* logσ² (log variance)

These parameters define a latent topic distribution.

```text
Document Embedding
      ↓
Encoder
      ↓
μ, logσ²
      ↓
Reparameterization
      ↓
Latent Vector z
      ↓
Topic Distribution θ
```

The model uses the reparameterization trick to sample latent topic representations during training.

---

### 5. Genre Prediction

The learned topic distribution is passed to a genre classifier.

```text
Topic Distribution θ
      ↓
Linear Classifier
      ↓
Genre Logits
      ↓
Genre Probabilities
```

Since genre prediction is a multi-label problem, the model uses a sigmoid activation and predicts multiple genres simultaneously.

---

### 6. Topic-Based Explanation

The model learns a topic-word matrix β.

Each row of β represents a latent topic and contains weights for words in the vocabulary.

```text
Topic Distribution θ
      ↓
Topic-Word Matrix β
      ↓
Topic Words
```

The highest-weighted words in each topic are used to explain model predictions.

### Important Note

The displayed topic words are **not extracted from the current input summary**.

Instead, they are derived from the learned topic-word matrix β that was trained on the entire corpus.

Therefore, topic words represent the model's learned understanding of a topic across the training data rather than keywords copied from the current input.

---

## Training Objective

The model is trained using a combination of:

### Genre Classification Loss

Binary Cross Entropy with Logits:

```text
BCEWithLogitsLoss
```

### KL Divergence Loss

Encourages the latent topic distribution to follow a smooth prior distribution.

```text
KL(q(z|x) || p(z))
```

### Topic Reconstruction Loss

The topic distribution is used to reconstruct word distributions through the learned topic-word matrix.

---

## Explainability

For each predicted genre, the application identifies:

1. The most influential topics for that genre.
2. The top words associated with those topics.
3. The contribution of each topic to the final prediction.

Example:

```text
Fantasy (0.91)

Topic 12:
magic, wizard, kingdom, dragon, spell, castle

Topic 7:
quest, hero, battle, sword, journey, war
```

This provides an interpretable explanation of why a particular genre was predicted.

---

## Technologies Used

* Python
* PyTorch
* Transformers (RoBERTa)
* Scikit-learn
* Streamlit
* Pandas
* NumPy

---

## Running the Application

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Repository Structure

```text
.
├── app.py
├── model.py
├── checkpoint.pt
├── mlb.pkl
├── vectorizer.pkl
├── requirements.txt
└── Notebooks/
```

---

## Future Improvements

* Topic contribution visualization
* Attention-based explanation methods
* Topic coherence evaluation
* Interactive topic exploration
* Lightweight deployment using compressed checkpoints

```
```
