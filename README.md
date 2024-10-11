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

## Custom Dataset Class

The `CustomDataset` class is designed to hold and structure datasets in a convenient manner. It allows you to specify:

- **Name**: Identifier for the dataset.
- **Train**: A list of Pandas DataFrames containing training data with columns `["text", "label"]`.
- **Test**: A list of Pandas DataFrames for testing data, also structured with `["text", "label"]`.
- **Label List**: A list of possible labels for classification.

-This class provides flexibility, allowing you to use any dataset you prefer, including your own. Note that no preprocessing of the text is done at this stage; this will be handled later when loading the text.
## HARD Dataset

The HARD dataset is loaded from a balanced reviews text file using the following code:

```python
df_HARD = pd.read_csv("/content/balanced-reviews.txt", sep="\t", header=0, encoding='utf-16')
df_HARD = df_HARD[["review", "rating"]]  # Focus on review and rating only
df_HARD.columns = [DATA_COLUMN, LABEL_COLUMN]
```
The ratings are converted into labels, where ratings greater than 3 are coded as 'POS' (positive), and ratings less than 3 are coded as 'NEG' (negative). The dataset is then split into training and testing sets.
```python
hard_map = {
    5: 'POS',
    4: 'POS',
    2: 'NEG',
    1: 'NEG'
}
```
This results in a balanced dataset with the following label distribution:

NEG: 52,164 reviews
POS: 53,899 reviews

This dataset is structured and stored using the CustomDataset class.

## LABR Dataset Processing

The `LABR` class is designed to handle the processing of Arabic book reviews from raw TSV files. It includes methods for:

- **Cleaning Reviews**: Removing URLs, HTML tags, and unwanted characters to produce a sanitized text corpus.
- **Reading Reviews**: Loading ratings, review IDs, user IDs, book IDs, and review bodies from clean and raw review files.
- **Data Splitting**: Creating training and testing datasets for multi-class (ratings 1-5) and binary classification (positive/negative).

This class facilitates the preparation of the LABR dataset for machine learning tasks.

## ArSAS Dataset Processing

The ArSAS dataset contains Arabic tweets labeled with sentiment. The data is read from a tab-separated file and filtered to include only the tweet text and sentiment label. The unique sentiment labels are: Positive, Negative, Neutral, and Mixed.

- **Total Tweets**: 19,897
- **Label Distribution**:
  - Negative: 7,384
  - Neutral: 6,894
  - Positive: 4,400
  - Mixed: 1,219

The dataset is split into training (15,917) and testing (3,980) sets.

## English Dataset Processing

The English dataset is read from a CSV file containing tweets, with columns for label and tweet text. The dataset is shuffled and labeled as follows: 

- **Label Mapping**: 
  - 0 → 0
  - 4 → 1

The dataset is divided into training, validation, and testing sets with 70%, 15%, and 15% of the data, respectively. The final structure includes the columns "text", "label", and a language indicator set to "english".

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

## Training Procedure

1. **Libraries**: The training process utilizes libraries such as `torch`, `transformers`, and `arabert`. Metrics like accuracy, F1 score, precision, and recall are computed using `sklearn`.

2. **Datasets**: The available datasets include `HARD`, `ASTD-Unbalanced`, and `ArSAS`. The `ArSAS` dataset is selected for training.

3. **Model Selection**: The model `aubmindlab/bert-base-arabertv02-twitter` is chosen from Hugging Face for Arabic text classification.

4. **Preprocessing**: The `ArabertPreprocessor` is used to preprocess the text data for both training and testing datasets.

5. **Tokenization**: Token lengths are analyzed to determine the maximum sentence length, with a limit set to 100 tokens to avoid truncation.

6. **Visualization**: Histograms of sentence lengths for training and testing datasets are plotted to visualize token distribution.

7. **Optimizer**: We use the AdamW optimizer, which adjusts learning rates during training to enhance convergence.
  
8. **Loss Function**: The Cross-Entropy Loss function calculates the difference between predicted outputs and true labels, guiding the model during training.

9. **Epochs**: Typically, 3 to 5 epochs are sufficient for training BERT, meaning the model passes over the entire dataset 3 to 5 times.

### Regular Training

## Training Parameters

The `TrainingArguments` class configures the training process. Key parameters include:
- **output_dir**: Directory for saving model checkpoints.
- **adam_epsilon**: Epsilon value for the Adam optimizer.
- **learning_rate**: Set to 2e-5 for fine-tuning.
- **fp16**: Enable mixed precision training for GPUs with tensor cores.
- **per_device_train_batch_size**: Batch size during training.
- **per_device_eval_batch_size**: Batch size during evaluation.
- **gradient_accumulation_steps**: Accumulate gradients over several batches.
- **num_train_epochs**: Total number of training epochs.
- **evaluation_strategy**: Set to 'epoch' for evaluations at the end of each epoch.
- **load_best_model_at_end**: Load the best model based on specified metrics.

### Model Training

The training is initiated using the `trainer.train()` method, and the results for each epoch, including training loss, validation loss, macro F1 score, accuracy, macro precision, and macro recall, are logged.
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

### Model Saving

After training, the model, tokenizer, and configuration are saved to a specified directory. The label mappings are also stored to facilitate future inference tasks.
## Predicting Sentiment with the Saved Model

To perform sentiment analysis using the trained model, you can utilize the `pipeline` function from the Transformers library. Below is a sample code snippet for making predictions:

### Code Snippet

```python
from transformers import pipeline

# Initialize the sentiment analysis pipeline
pipe = pipeline("sentiment-analysis", model="output_dir", device=0)

# Make a prediction
result = pipe("الفلم يبدو مقلق نوعا ما")
print(result)

### Copying the Model

Finally, the model is copied to Google Drive for easy access.

For more details on `TrainingArguments`, refer to the [Hugging Face documentation](https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments).

```
## Output Explanation
The output will provide the predicted sentiment along with associated scores for each category (e.g., Positive, Negative, Neutral, Mixed). For example:
```
[[{'label': 'Positive', 'score': 0.0022},  {'label': 'Negative', 'score': 0.9817},  {'label': 'Neutral', 'score': 0.0063},  {'label': 'Mixed', 'score': 0.0097}]]
```
In this case, the model predicts a Negative sentiment with a high confidence score of approximately 98.17%.


## 4. Testing and Evaluating the Model

After training, the model's performance is tested on a separate dataset that it hasn’t seen before. This ensures that the model generalizes well and can accurately predict the sentiment of new, unseen texts.

## Dataset Creation and Model Initialization

### Dataset Preparation

The `ClassificationDataset` class manages the text and label data for classification tasks. It accepts lists of training text and target labels, along with parameters for the model name, maximum sequence length, and a label mapping. The tokenizer (from the Hugging Face library) processes the input sentences, ensuring they fit within the defined maximum length of 256 tokens. Padding and truncation are applied as necessary to maintain uniform input sizes.

### Model Initialization

The `model_init` function is responsible for loading a pre-trained model for sequence classification. It uses `AutoModelForSequenceClassification` from Hugging Face and configures it to handle the number of labels defined in the label mapping.

### Evaluation Metrics

To evaluate model performance, the `compute_metrics` function calculates several key metrics: 
- **Accuracy**: Overall correctness of predictions.
- **Macro F1 Score**: Balances precision and recall across all classes.
- **Macro Precision**: Measures the correctness of positive predictions.
- **Macro Recall**: Assesses the ability to find all positive instances.

These metrics help in understanding how well the model is performing across different categories.

### Example Evaluation Code

```python
from sklearn.metrics import accuracy_score, f1_score

predictions = model.predict(test_data)

accuracy = accuracy_score(test_labels, predictions)
f1 = f1_score(test_labels, predictions, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"F1-Score: {f1}")
```
## K-Fold Cross-Validation

This section demonstrates how to perform K-fold cross-validation to evaluate the model's performance and optimize hyperparameters. 

### Steps

1. **Divide the Dataset**: Split the training set into K-folds for training and validation.
2. **Define Stratified K-Folds**: Utilize the `StratifiedKFold` class from `sklearn.model_selection`.

### Code Snippet

```python
from sklearn.model_selection import StratifiedKFold

# Define the number of folds
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

all_results = []

# Iterate through each fold
for fold_num, (train, dev) in enumerate(kf.split(kfold_dataset, kfold_dataset['label'])):
    # Prepare datasets for training and validation
    train_dataset = ClassificationDataset(...)
    val_dataset = ClassificationDataset(...)

    # Define training arguments
    training_args = TrainingArguments(...)

    # Initialize and train the model
    trainer = Trainer(...)
    trainer.train()

    # Evaluate and save the model
    results = trainer.evaluate()
    all_results.append(results)
    trainer.save_model(f"./train_{fold_num}/best_model")
```
# Ensemble all the cross validation models

This project employs an ensemble approach to sentiment analysis using multiple models trained on a labeled dataset. The models leverage the Hugging Face `transformers` library for state-of-the-art natural language processing capabilities.

## Overview

The purpose of this script is to predict sentiment labels (Negative, Neutral, Positive, Mixed) on a test set using the predictions from five different models trained with cross-validation. The ensemble method combines the predictions from these models to improve accuracy.

## Steps

1. **Load Required Libraries**: The script imports necessary libraries, including `transformers`, `pandas`, and `more_itertools`.

2. **Load the Label Map**: The inverse label map is created to decode model outputs back to human-readable labels.

3. **Load the Test Dataset**: The test set is extracted from the selected dataset for inference.

4. **Model Inference**:
   - A loop iterates through five models (assumed to be previously trained and saved).
   - Each model is used to perform sentiment analysis on the test set in batches for efficiency.
   - The predictions from each model are stored in a DataFrame.

5. **Ensemble Predictions**:
   - For each instance in the test set, scores from each model are aggregated.
   - The average score for each sentiment class is computed, and the class with the highest average score is selected as the final prediction.

6. **Results**:
   - The script outputs the count of predictions for each sentiment class.
   - A classification report is generated, showing precision, recall, and F1-score for each class.

## Output

The output of the script includes:
- A count of final predictions across sentiment classes:
  - **Negative**: 1603
  - **Neutral**: 1433
  - **Positive**: 932
  - **Mixed**: 12
- A classification report that provides detailed metrics for model performance.

## Conclusion

This ensemble method improves the robustness of sentiment predictions by leveraging the strengths of multiple models, enhancing overall accuracy and reliability.

   
## Challenges and Limitations

-**1. Multilingual Support**: Handling multiple languages adds complexity because different languages have different grammatical structures, vocabularies, and context usage. BERT's multilingual capability helps mitigate this, but further improvements could be made by fine-tuning it on larger, more diverse multilingual datasets.


-**2. Data Preprocessing**: Text data often contains noisy or unstructured elements (e.g., misspellings, slang, emojis) that can impact model performance. Proper cleaning and tokenization are critical for ensuring accurate predictions.

### How to Run the Project

-**1. Clone the repository**: Download the code by running the following command:

```git clone https://github.com/yourusername/multi-lang-sentiment-analysis.git
cd multi-lang-sentiment-analysis
```

-**2. Install Dependencies**: Make sure you have all the required packages installed by running:
```
pip install -r requirements.txt
```

-**3. Run the Notebook**: Open Sentiment_Analysis_BERT-uncased.ipynb in Jupyter Notebook and follow the instructions for data preprocessing, training, and evaluation.

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

