## Multi-Language Sentiment Analysis using BERT


## Overview
This project aims to perform sentiment analysis across multiple languages using BERT, a deep learning model. Sentiment analysis identifies whether text conveys positive, negative, or neutral sentiment. BERT's multilingual capabilities make it suitable for global applications.

## Why BERT?
BERT processes text bidirectionally, considering both preceding and following words in a sentence. This enables it to capture context and nuances effectively, making it powerful for tasks like sentiment analysis.

## Project Components

The project has four main steps:

1. **Loading and Preprocessing the Data**
   
2. **Preparing the BERT Model**

3. **Training the Model**

4. **Testing and Evaluating the Model**

## 1. Loading and Preprocessing the Data

Data preprocessing is one of the most crucial steps in any NLP task. In this project, we start by loading a CSV file that contains text data and corresponding sentiment labels (positive, negative, or neutral). Each row of the CSV file includes a sentence and a sentiment classification.

**Example Data Format:**


The data must be cleaned and tokenized before being fed into the model. Tokenization is the process of breaking down a sentence into smaller components called tokens. In BERT, this step converts text into a format that can be understood by the model.

**Example Tokenization Process:**

```python
from transformers import BertTokenizer

# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize a sample text
sample_text = "This is an amazing project!"
tokens = tokenizer(sample_text, padding=True, truncation=True, return_tensors="pt")
print(tokens)
```
## 2. Preparing the BERT Model

We use HuggingFace Transformers, a popular NLP library, to load a pre-trained BERT model (`bert-base-uncased`). This version of BERT has been trained on large datasets of text and is uncased, meaning it treats uppercase and lowercase letters the same. Fine-tuning this model for sentiment analysis allows us to leverage its extensive pre-training for our specific task.

### Preparing the Model

Once the data is tokenized, it can be passed into the BERT model, which outputs hidden representations for each token. These representations are then fed into a classifier to predict the sentiment.

### Model Architecture

The architecture consists of:

- **BERT Layers**: The core part of the model that processes tokenized inputs and generates context-aware representations of each token.
- **Classification Layer**: A fully connected layer that maps the output of BERT to sentiment classes (positive, negative, neutral).

**Example Code:**

```python
from transformers import BertForSequenceClassification

# Load pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
```

## 3. Training the Model

Once the data is preprocessed and the model is set up, we begin the training process. Fine-tuning BERT involves modifying its internal weights to adjust for the specific sentiment analysis task. During training, we adjust these weights based on the provided data.

### Training Setup

- **Optimizer**: We use the AdamW optimizer, which adjusts learning rates during training to enhance convergence.
  
- **Loss Function**: The Cross-Entropy Loss function calculates the difference between predicted outputs and true labels, guiding the model during training.

- **Epochs**: Typically, 3 to 5 epochs are sufficient for training BERT, meaning the model passes over the entire dataset 3 to 5 times.

### Training Loop Example

```python
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=2e-5)

for epoch in range(epochs):
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

## 4. Testing and Evaluating the Model

After training, the model's performance is tested on a separate dataset that it hasn’t seen before. This ensures that the model generalizes well and can accurately predict the sentiment of new, unseen texts.

### Evaluation Metrics

- **Accuracy**: Measures the percentage of correctly predicted labels.
- **F1-Score**: A weighted measure of precision and recall that evaluates how well the model handles imbalanced classes.

### Example Evaluation Code

```python
from sklearn.metrics import accuracy_score, f1_score

predictions = model.predict(test_data)

accuracy = accuracy_score(test_labels, predictions)
f1 = f1_score(test_labels, predictions, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"F1-Score: {f1}")
```
### Results

- **Accuracy**: 85%
- **F1-Score**: 84%
These results indicate that the fine-tuned BERT model performs well in predicting sentiments across multiple languages, but improvements can still be made.
   
## Challenges and Limitations

### 1. Multilingual Support: Handling multiple languages adds complexity because different languages have different grammatical structures, vocabularies, and context usage. BERT's multilingual capability helps mitigate this, but further improvements could be made by fine-tuning it on larger, more diverse multilingual datasets.


### 2. Data Preprocessing: Text data often contains noisy or unstructured elements (e.g., misspellings, slang, emojis) that can impact model performance. Proper cleaning and tokenization are critical for ensuring accurate predictions.

### How to Run the Project

## 1. Clone the repository: Download the code by running the following command:

```git clone https://github.com/yourusername/multi-lang-sentiment-analysis.git
cd multi-lang-sentiment-analysis
```


## 2. Install Dependencies: Make sure you have all the required packages installed by running:
```
pip install -r requirements.txt
```

## 3. Run the Notebook: Open Sentiment_Analysis_BERT-uncased.ipynb in Jupyter Notebook and follow the instructions for data preprocessing, training, and evaluation.

### Future Improvements

-**Expand Language Support**: Include more languages to increase the diversity of the dataset and enhance model robustness.

-**Data Augmentation**: Introduce techniques like back-translation to generate additional data and improve model generalization.

-**Hyperparameter Tuning**: Experiment with different learning rates, batch sizes, and training strategies to optimize the model’s performance.

### Acknowledgments

-**HuggingFace**: For providing the Transformers library, which simplifies the process of using BERT and other pre-trained models.

-**PyTorch**: For offering a flexible deep learning framework used to build and train the model.

-**The Data Community**: For contributing to open datasets that enable sentiment analysis across multiple languages.

## Datasets:

* https://huggingface.co/datasets/tyqiangz/multilingual-sentiments


* https://www.kaggle.com/datasets/weywenn/sentiment-analysis-multilanguage/data

* https://paperswithcode.com/paper/xed-a-multilingual-dataset-for-sentiment

* https://github.com/facebookresearch/XNLI


## Recommanded Models

* https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment

