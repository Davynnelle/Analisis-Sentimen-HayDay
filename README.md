**Sentiment Analysis Pipeline for Google Play Reviews**

This repository contains an end-to-end pipeline to scrape, preprocess, label, visualize, and model sentiment analysis on Indonesian reviews from the Google Play Store using deep learning architectures (LSTM, BiLSTM, GRU).

---

## Table of Contents
1. [Features](#features)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Project Structure](#project-structure)
5. [Usage](#usage)
   - [1. Data Scraping](#1-data-scraping)
   - [2. Data Loading](#2-data-loading)
   - [3. Preprocessing](#3-preprocessing)
   - [4. Sentiment Labeling](#4-sentiment-labeling)
   - [5. Visualization](#5-visualization)
   - [6. Model Preparation & Training](#6-model-preparation--training)
   - [7. Model Evaluation](#7-model-evaluation)
   - [8. Inference](#8-inference)
6. [Results & Conclusion](#results--conclusion)
7. [References](#references)

---

## Features
- Scraping up to 60,000 Indonesian reviews from Google Play Store using `google-play-scraper` and `langdetect`.
- Comprehensive text cleaning (emoji removal, URL/hashtag/user mention stripping, slang normalization, casing, punctuation removal).
- Language-specific stemming and stopword removal with **Sastrawi** and **NLTK**.
- Rule-based sentiment labeling via positive/negative lexicons with negation handling.
- Exploratory visualizations: score distributions, word clouds, class percentages, top-10 word frequencies.
- Experiments with three deep-learning classifiers: LSTM, BiLSTM, and GRU.
- Model training with callbacks: `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`, and custom accuracy-based stop.
- Performance comparison and inference function with confidence scores.

---

## Requirements
- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- wordcloud
- google-play-scraper
- tensorflow
- transformers
- sastrawi
- emoji
- tqdm
- langdetect
- nltk

You can install all dependencies with:

```bash
pip install google-play-scraper pandas numpy matplotlib seaborn wordcloud tensorflow transformers sastrawi emoji tqdm nltk langdetect
```


---

## Project Structure

```
├── README.md
├── scrape_and_preprocess.py    # Scraping & cleaning pipeline
├── label_sentiment.py          # Lexicon-based labeling
├── visualize.py                # EDA & plots
├── train_models.py             # Model definitions & training experiments
├── evaluate.py                 # Model evaluation & comparison
├── inference.py                # Inference function & examples
├── hayday_reviews.csv          # Scraped raw data (output)
├── hasil_label_sentimen.csv    # Labeled dataset (output)
└── saved_models/               # Saved model checkpoints
```

---

## Usage

Clone the repo and navigate into the directory:

```bash
git clone https://github.com/yourusername/indonesia-play-review-sentiment.git
cd indonesia-play-review-sentiment
```

### 1. Data Scraping

Run `scrape_and_preprocess.py` to:
- Scrape reviews for `com.supercell.hayday` (60k by default) filtered by lang=`id`.
- Filter non-Indonesian text via `langdetect`.
- Remove duplicates and save to `hayday_reviews.csv`.

```bash
python scrape_and_preprocess.py --app_id com.supercell.hayday --target_reviews 60000
```

### 2. Data Loading

Load the CSV into a DataFrame for downstream steps:

```python
from label_sentiment import load_dataset

df = load_dataset('hayday_reviews.csv')
```

### 3. Preprocessing

The `EnhancedGamePreprocessor` class handles:
1. Case folding, emoji removal, URL/hashtag/user cleaning.
2. Punctuation and digit removal.
3. Slang normalization via a custom dictionary.
4. Stopword removal (Indonesian, English, custom) while preserving negations.
5. Stemming with Sastrawi.

Use it to clean and tokenize:

```python
from scrape_and_preprocess import EnhancedGamePreprocessor

pre = EnhancedGamePreprocessor()
df['clean_text'] = df['content'].apply(pre.clean)
```

### 4. Sentiment Labeling

`label_sentiment.py` fetches positive/negative lexicons from GitHub and labels each review:

```python
from label_sentiment import sentiment_analysis_lexicon

score, label = sentiment_analysis_lexicon(df['clean_text'].iloc[0])
```

Results stored in `hasil_label_sentimen.csv` with columns: `labeling_score`, `labeling`.

### 5. Visualization

Run `visualize.py` to generate:
- Bar charts for original vs. labeled distributions.
- Word clouds per sentiment class.
- Pie chart of class percentages.
- Bar chart of top-10 words.

```bash
python visualize.py
```

### 6. Model Preparation & Training

`train_models.py` defines three experiments:
1. **LSTM** (80/20 split, 200-length sequences)
2. **BiLSTM** (70/30 split, 100-length)
3. **GRU** (80/20, 200-length)

Each uses embedding layers, dropout, batch normalization, and trains with callbacks. Outputs saved in `saved_models/`.

```bash
python train_models.py
```

### 7. Model Evaluation

Use `evaluate.py` to:
- Generate confusion matrices and classification reports.
- Plot validation-accuracy comparisons.
- Summarize train vs. test accuracy in a DataFrame.

```bash
python evaluate.py
```

### 8. Inference

Run `inference.py` for new texts:

```bash
python inference.py
```

Or use the `predict_sentiment` function:

```python
from inference import predict_sentiment

result = predict_sentiment(
    "Game ini seru banget!", model=bilstm_model,
    tokenizer=tokenizer, preprocessor=pre, le=label_encoder,
    max_length=100
)
print(result)
```

---

## Results & Conclusion

- **BiLSTM** achieved the best test accuracy (94%) with strong generalization (train 96.1%).
- **LSTM** reached 96.0% train and balanced for test accuracy (93%).
- **GRU** balanced performance (test 92%).
- Lexicon-based labeling provides quick bootstrapping but can mis-handle complex negations.

Overall, the pipeline demonstrates an effective workflow to analyze Indonesian app reviews using both rule-based and DL methods.

---

## References

- [google-play-scraper](https://github.com/facundoolano/google-play-scraper)
- [Sastrawi](https://github.com/sastrawi/sastrawi)
- [NLTK](https://www.nltk.org/)
- [TensorFlow Keras](https://www.tensorflow.org/guide/keras)
- [FastText Indonesian Word Vectors](https://fasttext.cc/) (optional)

📄 Note: This project is based on an final submission from the course "Belajar Pengembangan Machine Learning" on Dicoding Indonesia. The goal was to implement and expand on the core concepts introduced in the course.
