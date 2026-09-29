# Text Emotion Classification with Bidirectional GRU

**NLP · Deep Learning · TensorFlow/Keras · Model Comparison**

This project classifies a text passage into one of six emotions: **sadness, joy, love, anger, fear, or surprise**. It compares recurrent neural network architectures and uses a stacked bidirectional GRU as the final model.

## Project overview

| Component | What the notebook does |
| --- | --- |
| Dataset | Uses the `dair-ai/emotion` text classification dataset: 16,000 training examples and a 2,000-example test split |
| Exploration | Checks missing values and visualizes the training label distribution |
| Text preparation | Fits a tokenizer on training text, limits the vocabulary to 10,000 tokens, and pads/truncates sequences to 50 tokens |
| Class imbalance | Applies balanced class weights during training |
| Architectures | Compares stacked Simple RNN, LSTM, and GRU models, then trains a stacked Bidirectional GRU with a 300-dimensional learned embedding and dropout |
| Evaluation | Records accuracy and loss, displays a confusion matrix, and tests five sample sentences |
| Artifacts | Exports the final Keras model and tokenizer for potential later integration |

## Results recorded in the notebook

| Model | Reported accuracy |
| --- | ---: |
| Simple RNN | 26.45% |
| LSTM | 33.95% |
| GRU | 28.90% |
| **Bidirectional GRU** | **92.10%** |

The Bidirectional GRU also records a loss of **0.2258** on the dataset's test split. The sample predictions show four expected emotions and one apparent error: a clearly happy sentence was predicted as *fear*. This makes error analysis useful even with strong aggregate accuracy.

**Evaluation caveat:** the notebook supplies the dataset's test split as `validation_data` during training and monitors its validation loss for early stopping. The reported 92.10% is therefore performance on a split consulted for model selection, rather than a fully untouched final test result. A separate validation split and untouched test evaluation would make the estimate stronger.

## Skills demonstrated

Text preprocessing, sequence modeling, class weighting, recurrent architecture comparison, TensorFlow/Keras training, confusion-matrix analysis, and model serialization.

**Project notebook:** `final_clean.ipynb`  
**Dataset:** [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion)
