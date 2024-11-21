
# **Arabic Sentiment Analysis using BERT**

---

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Getting Started](#getting-started)
4. [Dependencies and Setup](#dependencies-and-setup)
5. [Datasets](#datasets)
6. [Notebook Workflow](#notebook-workflow)
7. [Usage](#usage)
8. [Results and Evaluation](#results-and-evaluation)
9. [Future Work](#future-work)
10. [Acknowledgments](#acknowledgments)

---

## **Project Overview**
This project explores sentiment analysis for Arabic text using BERT-based models, focusing on fine-tuning pre-trained models like AraBERT on Arabic sentiment datasets. Sentiment analysis is a crucial task in natural language processing (NLP), enabling automated understanding and classification of opinions expressed in text. This project tackles unique challenges posed by Arabic language processing, such as its complex morphology and script.

---

## **Features**
- **Preprocessing**: Tokenization and cleaning of Arabic text, including diacritic removal and stemming using Farasa.
- **Model Training**: Fine-tuning a pre-trained BERT model for sentiment analysis.
- **Multi-Dataset Support**: Integrates multiple Arabic sentiment datasets for comprehensive evaluation.
- **GPU Optimization**: Utilizes CUDA for faster model training and inference.

---

## **Getting Started**
Follow these steps to set up and run the project:
1. Install dependencies.
2. Download required datasets.
3. Execute the notebook to preprocess data, fine-tune the model, and evaluate results.

---

## **Dependencies and Setup**
To set up your environment, install the following:

### Install Python Libraries
```bash
pip install -q -U transformers==4.12.2 datasets
pip install farasapy==0.0.14 pyarabic==0.6.14 emoji==1.6.1 sentencepiece==0.1.96
```

### Clone Necessary Repositories
```bash
# Clone AraBERT repository
git clone https://github.com/aub-mind/arabert

# Clone Arabic datasets
git clone https://github.com/elnagara/HARD-Arabic-Dataset
git clone https://github.com/mahmoudnabil/ASTD
git clone https://github.com/nora-twairesh/AraSenti
git clone https://github.com/mohamedadaly/AraSentiMent
```

### Hardware Requirements
- **GPU**: Recommended for training (e.g., NVIDIA GPU with CUDA support).
- **RAM**: At least 16 GB for handling datasets and model processing.

---

## **Datasets**
The following datasets are used for sentiment analysis:

1. **HARD-Arabic-Dataset**: A balanced dataset for sentiment classification.
2. **ASTD**: Arabic Social Media Dataset for fine-grained sentiment analysis.
3. **AraSenti**: A large-scale Arabic sentiment dataset.
4. **AraSentiMent**: Focuses on sentiment analysis in various Arabic dialects.

Each dataset undergoes preprocessing to standardize formats, remove noise, and tokenize using the AraBERT tokenizer.

---

## **Notebook Workflow**
The notebook is organized into clear sections:

### 1. **Dependency Installation**
   - Installs libraries such as `transformers`, `datasets`, `pyarabic`, and others.
   - Clones AraBERT and dataset repositories.

### 2. **Data Preparation**
   - **Cleaning**: Removes diacritics, punctuation, and unnecessary whitespace.
   - **Tokenization**: Applies AraBERT tokenizer to prepare text for model input.

### 3. **Model Configuration**
   - **Pre-trained Model**: Loads AraBERT from the `transformers` library.
   - **Hyperparameters**: Configures learning rate, batch size, and epochs.

### 4. **Training**
   - Fine-tunes AraBERT on the training split of the dataset.
   - Uses cross-entropy loss and Adam optimizer for gradient updates.

### 5. **Evaluation**
   - Computes accuracy, precision, recall, and F1-score on test data.
   - Outputs confusion matrices and other performance metrics.

### 6. **Visualization**
   - Training and validation loss curves.
   - Performance metrics in tabular and graphical formats.

---

## **Usage**
### Steps to Run:
1. Clone this repository and navigate to the notebook file.
2. Install all dependencies as mentioned above.
3. Run the notebook cell-by-cell to:
   - Preprocess datasets.
   - Train the AraBERT model.
   - Evaluate and visualize results.

For troubleshooting, ensure CUDA is enabled and check for library compatibility issues.

---

## **Results and Evaluation**
The model demonstrates strong performance in sentiment classification tasks:

### Key Metrics:
- **Accuracy**: Achieved high accuracy across multiple datasets.
- **F1-Score**: Balanced performance for both positive and negative classes.
- **Confusion Matrix**: Shows the distribution of correctly and incorrectly classified samples.

### Visualizations:
- Training progress and validation loss curves.
- Evaluation metrics presented in charts for clarity.

---

## **Future Work**
- Experiment with newer transformer architectures (e.g., RoBERTa, GPT-based models).
- Expand dataset coverage to include underrepresented dialects.
- Optimize preprocessing to handle noisy real-world text.

---

## **Acknowledgments**
This project leverages open-source tools and datasets provided by the NLP and Arabic language communities:
- AraBERT by the American University of Beirut (AUB).
- Datasets contributed by Arabic NLP researchers.

---
