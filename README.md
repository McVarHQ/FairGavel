# FairGavel: Leveraging Graph Neural Networks for Fairness-Driven Legal Judgment Prediction

<p align="center">
  <img src="./images/model pipeline.png" alt="FairGavel Pipeline" height="500">
  <br>
  <span style="font-size: 14px; color: gray;"><b>FairGavel Pipeline</b></span>
</p>

## 🔍 Overview

**FairGavel** is a fairness-driven, graph-based AI system that predicts legal verdicts from court documents using **Graph Neural Networks (GNNs)**. By modeling legal cases as structured graphs and incorporating adversarial debiasing, FairGavel ensures both high predictive accuracy and fairness across sensitive attributes (e.g., region, language, nationality). This project is based on the **MultiLexSum** dataset and is designed to promote responsible AI in the legal domain.

---

## 📜 Abstract

The application of Artificial Intelligence (AI) in the legal field has provided new means of automating intricate decision-making processes. However, maintaining **fairness** and **transparency** in AI-based legal judgment prediction remains largely unsolved.

FairGavel presents a **GNN-based system** that models legal cases from the MultiLexSum dataset as graphs of sentences and entities. By encoding legal documents with structural relationships and employing adversarial fairness modules, FairGavel not only achieves competitive performance but also demonstrates lower fairness leakage across sensitive attributes.

This system illustrates how **explainability**, **fairness**, and **high performance** can coexist in legal AI models.

---

## ⚙️ Core Components

### 1. MultiLexSum Dataset

The **MultiLexSum** dataset is a large-scale, multilingual legal corpus collected from Indian court cases. It includes full-length legal documents, summaries, and metadata such as region, language, court level, and protected attributes.

Key Highlights:
- Over 100K annotated legal cases
- Multiple languages (English, Hindi, etc.)
- Includes **verdict labels**, **summaries**, and **sensitive attributes**

This dataset enables **fairness-aware** training and supports both extractive and abstractive summarization tasks in addition to verdict prediction.

<p align="center">
  <img src="./images/6. Word Cloud - Combined Sources.png" alt="Word Cloud of MultiLexSum" height="300">
  <br>
  <span style="font-size: 14px; color: gray;"><b>Word Cloud of MultiLexSum</b></span>
</p>

---

### 2. Summarization & Preprocessing

- Multilingual tokenization and cleaning
- Optional summarization (long, short, tiny)
- Sensitive attribute identification (e.g., pronouns, key terms)

### 3. Graph Construction

- **Nodes**: Sentences, Entities (Plaintiff/Defendant)
- **Edges**: Sequential, Semantic Similarity, Entity-Sentence, Document-level, Self-loops
- Edge-type labels enable heterogeneous GNN processing

<p align="center">
  <img src="./images/newplot.png" alt="Sample Graph" height="300">
  <br>
  <span style="font-size: 14px; color: gray;"><b>Sample Graph</b></span>
</p>

### 4. FairGavel Model Architecture

<p align="center">
  <img src="./images/model architecture.png" alt="FairGavel Architecture" height="300">
  <br>
  <span style="font-size: 14px; color: gray;"><b>FairGavel Architecture</b></span>
</p>

- **Graph Encoder**: 2-layer GraphSAGE with gated attention pooling
- **Verdict Regressor**: Predicts verdict as a continuous score ∈ [0, 1]
- **Adversarial Debiasing Module**: Multiple heads for protected attributes (e.g., region, language, urbanicity) using Gradient Reversal Layer (GRL)

---

## 🧪 Training Configuration

| Parameter                    | Value         |
|-----------------------------|---------------|
| Input Channels              | 384           |
| Hidden Channels             | 128           |
| Layers                      | 2             |
| Neighbors per Layer         | [25, 10]      |
| Number of Subgraphs         | 6             |
| Learning Rate               | 0.001         |
| Weight Decay                | 1e-5          |
| Epochs                      | 20            |
| Neighbor Batch Size         | 32            |

---

## 📉 Loss Curves

Smoothed training losses across epochs:
- Total Loss
- Regression Loss
- Adversarial Loss
- Validation Loss

<p align="center">
  <img src="./images/1. All Losses with Zoom.png" alt="Loss Graph" height="300">
  <br>
  <span style="font-size: 14px; color: gray;"><b>Loss Graph</b></span>
</p>

Highlights:
- Best validation epoch: 15  
- Best fairness-accuracy balance: Epoch 8

---

## ✅ Performance & Fairness Metrics

| Metric                           | Value    |
|----------------------------------|----------|
| **Accuracy (Test)**              | 93.3%    |
| **False Positive Rate (FPR)**    | 8.3%     |
| **False Negative Rate (FNR)**    | 5.6%     |
| Final Regression Loss            | 0.0083   |
| Final Adversarial Loss           | 0.6637   |
| Reg/Adv Loss Ratio               | 0.0125   |
| Generalization Gap               | 0.0043   |

FairGavel achieves **state-of-the-art fairness** without sacrificing predictive performance, demonstrating the effectiveness of adversarially debiased GNNs.

---

## 📂 Repository Structure
fairgavel/
│
├── Codes/ # Project Codes
├── Data/ # Accessories and Intermediaries
├── Docs/ # Documentation
├── images/ # Diagrams and results for README
├── requirements.txt # Dependencies
└── README.md # You're here!
