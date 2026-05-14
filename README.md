# Quora Question Duplicate Detection using Hybrid Deep Learning Model

This project builds a Hybrid NLP Model to detect whether two questions are duplicates or not using:

- Deep Learning (BiLSTM)
- Machine Learning Features
- TF-IDF Similarity
- Fuzzy Matching
- NLP Text Preprocessing

The model is trained on the Quora Question Pairs Dataset.

---

# Features

## NLP Preprocessing
- Lowercasing
- HTML tag removal
- Contraction handling
- Punctuation removal
- Special character replacement

## Feature Engineering
- Character length
- Word count
- Common word count
- Word share ratio
- Fuzzy matching scores
- Cosine similarity using TF-IDF

## Deep Learning Architecture
- Shared BiLSTM Network
- Embedding Layer
- Feature Fusion
- Dense Neural Network
- Batch Normalization
- Dropout Regularization

---

# Tech Stack

- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- RapidFuzz
- Matplotlib

---

# Dataset

Dataset used:
Quora Question Pairs Dataset

Expected CSV columns:

- id
- qid1
- qid2
- question1
- question2
- is_duplicate

---

# Project Structure

```bash
├── train.csv
├── final_hybrid_modelsl.py
├── model.h5
├── model1.keras
├── tokenizer.pkl
├── tfidf.pkl
├── scaler.pkl
└── README.md
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/your-username/quora-question-duplicate-detection.git
cd quora-question-duplicate-detection
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Required Libraries

```bash
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn rapidfuzz
```

---

# Model Architecture

## Inputs
- Question 1 sequence
- Question 2 sequence
- Engineered numerical features

## Neural Network
- Embedding Layer
- Shared Bidirectional LSTM
- Feature concatenation
- Dense layers
- Sigmoid output

---

# Training

Run:

```bash
python final_hybrid_modelsl.py
```

---

# Saved Files

The following files are generated after training:

- `model.h5`
- `model1.keras`
- `tokenizer.pkl`
- `tfidf.pkl`
- `scaler.pkl`

---

# Evaluation Metrics

- Accuracy
- Log Loss
- F1 Score
- Confusion Matrix
- Classification Report

---

# Future Improvements

- Add Attention Mechanism
- Use Transformer Models (BERT)
- Hyperparameter tuning
- Deploy using Flask/FastAPI
- Create Streamlit Web App

---

# Author

Amit Kurmi

---

# License

This project is open-source and available under the MIT License.
