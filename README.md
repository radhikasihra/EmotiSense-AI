# 🧠 EmotiSense AI — Intelligent Text Emotion Classification

**NLP · Deep Learning · Bidirectional GRU · TensorFlow · Keras**

## 📌 Overview

**EmotiSense AI** is an NLP-based deep learning system designed to classify text into six emotions: **Sadness, Joy, Love, Anger, Fear, and Surprise**.

The project compares multiple recurrent neural network architectures — **Simple RNN, LSTM, GRU, and Bidirectional GRU** — and selects a **Stacked Bidirectional GRU** as the final model based on performance.

## 📊 Dataset

The project uses the **`dair-ai/emotion`** text classification dataset.

- **Training samples:** 16,000
- **Test samples:** 2,000
- **Emotion classes:** 6
- **Vocabulary size:** 10,000 tokens
- **Sequence length:** 50 tokens

## ⚙️ AI/ML Pipeline

**Text Data → Preprocessing → Tokenization → Sequence Padding → Class Weighting → Deep Learning Models → Model Comparison → Emotion Prediction**

Text preprocessing includes tokenization and sequence padding/truncation. **Balanced class weights** are applied during training to address class imbalance.

## 🧠 Model Architecture

The final **EmotiSense AI** model uses a **Stacked Bidirectional GRU** architecture with:

- **300-dimensional learned embeddings**
- **Bidirectional GRU layers**
- **Dropout regularization**
- **Class-weighted training**
- **Early stopping**
- **Multi-class emotion classification**

The Bidirectional architecture processes textual sequences in both directions, enabling the model to capture contextual relationships more effectively.

## 📈 Model Performance

| Model | Accuracy |
| --- | ---: |
| Simple RNN | 26.45% |
| LSTM | 33.95% |
| GRU | 28.90% |
| **Bidirectional GRU (EmotiSense AI)** | **92.10%** |

### 🏆 Final Model

**Accuracy: 92.10%**  
**Loss: 0.2258**

The **Bidirectional GRU significantly outperformed the baseline RNN, LSTM, and GRU architectures**, making it the selected model for EmotiSense AI.

## 🔍 Model Evaluation

Model performance is analyzed using:

- Accuracy and loss
- Confusion matrix
- Architecture comparison
- Sample emotion predictions
- Prediction error analysis

> **Evaluation Note:** The reported 92.10% accuracy comes from the dataset test split used as validation data during training and early stopping. A separate validation set and untouched test set would provide a stronger final evaluation.

## 🛠️ Tech Stack

**Programming:** Python  
**Deep Learning:** TensorFlow, Keras  
**NLP:** Tokenization, Sequence Padding, Text Preprocessing  
**Architectures:** Simple RNN, LSTM, GRU, Bidirectional GRU  
**ML Techniques:** Class Weighting, Dropout Regularization, Early Stopping  
**Evaluation:** Accuracy, Loss, Confusion Matrix

## 💡 Skills Demonstrated

- Natural Language Processing (NLP)
- Deep Learning
- Text Classification
- Sequence Modeling
- Bidirectional GRU
- TensorFlow & Keras
- Text Preprocessing
- Class Imbalance Handling
- Model Training & Evaluation
- Deep Learning Model Comparison

## 🎯 Project Outcome

**EmotiSense AI** demonstrates an end-to-end NLP and deep learning workflow for **multi-class emotion recognition**, achieving **92.10% reported accuracy** with a Stacked Bidirectional GRU while substantially outperforming the baseline recurrent architectures.
