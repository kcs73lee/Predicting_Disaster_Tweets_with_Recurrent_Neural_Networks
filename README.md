# Predicting Disaster Tweets with Recurrent Neural Networks

## Introduction

Twitter is a powerful platform for real-time updates during emergencies—but not every tweet using dramatic language is about a real disaster. This project aims to classify whether a tweet refers to an actual disaster (`1`) or not (`0`) using natural language processing (NLP) and deep learning techniques.

We compare multiple model architectures:
- **DistilBERT** via Ktrain (low-code wrapper)
- **Recurrent Neural Networks (RNN)** including:
  - Long Short-Term Memory (LSTM)
  - Gated Recurrent Unit (GRU)

RNNs are well-suited for this task due to their ability to process sequences and capture context over time.

---

## Dataset

From the [Kaggle competition](https://www.kaggle.com/competitions/nlp-getting-started/data), the dataset includes:
- `train.csv`: tweets and disaster labels
- `test.csv`: tweets for prediction
- `sample_submission.csv`: format for Kaggle submission

Key columns:
- `id`: unique identifier
- `text`: tweet content
- `keyword`: keyword used in tweet
- `location`: location of tweet
- `target`: disaster label (1 or 0, only in training set)

---

## Project Workflow

- **Environment Setup**  
  Import libraries (TensorFlow, sklearn) and configure GPU

- **EDA & Preprocessing**  
  Clean missing data, drop duplicates, and tokenize text

- **Model Architectures**
  - **DistilBERT (Ktrain)**
  - **LSTM (Keras)**
  - **GRU (Keras)**

- **Hyperparameter Tuning (LSTM)**  
  Add bidirectional layers, dropout, normalization, and tune early stopping

- **Evaluation**  
  Assess models using loss and accuracy metrics

- **Prediction & Submission**  
  Generate test set predictions using the best model and submit to Kaggle

---

## Summary

This project demonstrates how different deep learning models perform on disaster tweet classification. Final reflections include model strengths, limitations, and possible improvements.

---
