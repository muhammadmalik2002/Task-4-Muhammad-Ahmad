# Project 4: NLP & Sentiment Analysis

**Data Science Internship — Decode Labs (Optional Mastery Phase)**
**Intern:** Muhammad Ahmad

## Overview
This project builds an end-to-end NLP pipeline that converts unstructured movie review text into a mathematical representation using TF-IDF, and trains machine learning classifiers to predict sentiment (Positive/Negative).

## Dataset
- **Source:** [Kaggle — IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Size:** 50,000 movie reviews — perfectly balanced (25,000 positive / 25,000 negative)
- **Fields:** `review` (raw text), `sentiment` (positive/negative label)

> Note: dataset file (`IMDB Dataset.csv`) is not included in this repo due to size. Download it from the link above.

## What Was Done

### 1. Text Pre-Processing Pipeline
- Removed HTML tags (`<br>`) from raw scraped review text
- Lowercased and stripped non-alphabetic characters
- Tokenized using NLTK's `word_tokenize`
- Removed stop-words — **with negation words explicitly preserved** (`not`, `no`, `never`, etc.), since default stop-word removal would strip "not" and invert sentence meaning (e.g. "not good" → "good")
- Applied **POS-guided lemmatization** (NLTK `WordNetLemmatizer`), passing part-of-speech tags for correct morphological reduction (e.g. "went" → "go")

### 2. Vectorization
- **TF-IDF** with unigrams + bigrams (to capture negated phrases like "not good")
- Vocabulary capped at 10,000 features, `min_df=2` to exclude noise/typos

### 3. Model Training
- 80/20 stratified train/test split
- Trained and compared **Multinomial Naive Bayes** (Laplace smoothing) and **Linear SVM**

## Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Naive Bayes | 86.93% | 0.8561 | 0.8878 | 0.8716 |
| **SVM (Linear)** | **88.91%** | **0.8880** | **0.8904** | **0.8892** |

**SVM outperformed Naive Bayes across every metric** and produced a more balanced confusion matrix with fewer total misclassifications (1,109 vs. 1,307).

### Sample Predictions
| Review | Prediction |
|---|---|
| "This movie was absolutely fantastic, I loved every minute of it!" | Positive ✅ |
| "Terrible film, complete waste of time. Not good at all." | Negative ✅ |
| "It was okay, not the best but not the worst either." | Negative (reasonable — genuinely mixed/neutral review) |

## Repository Contents
- Project notebook (Google Colab compatible)
- `Project 4 Report.docx` — full methodology and results
- Original project brief from Decode Labs

## Tools & Libraries
`Python`, `NLTK` (tokenization, stop-words, WordNetLemmatizer, POS tagging), `Scikit-learn` (TfidfVectorizer, MultinomialNB, LinearSVC)

## How to Run
1. Download `IMDB Dataset.csv` from the [Kaggle link](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and upload it to Google Drive
2. Open the notebook in Google Colab
3. Update the `DATA_PATH` variable to point to your uploaded file
4. Run all cells top to bottom (full preprocessing takes a few minutes on the 50k dataset)
