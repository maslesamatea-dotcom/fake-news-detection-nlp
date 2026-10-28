# Fake News Detection (NLP + Machine Learning)

**Language:** English | [Srpski](README.sr.md)

Course project, Artificial Intelligence
- **Author:** Matea Masleša
- **Course instructor:** Dr Đorđe Petrović
- **School:** Information Technology School (ITS), Belgrade, June 2026

## Goal

Binary text classification: is a news article **fake** or **real**. TF-IDF vectorisation and three classifiers are compared: Logistic Regression, Naive Bayes and Random Forest.

## Data

- **Source:** Kaggle, *Fake and Real News Dataset* ([link](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)). The data are not in the repository (see `data/README.md`).
- **Size:** 23,481 fake and 21,417 real articles (44,898 in total), in English, mostly political topics from 2015–2018 with a focus on the 2016 US election.
- **Columns:** title, text, category and date. Category and date are not used as features; the model learns from the text only.
- **Dataset cleaning:** 6,252 duplicates (by text) were removed (38,646 remaining), then records with text shorter than 100 or title shorter than 5 characters. The final set has **38,343** articles: 30,674 for training (13,721 fake, 16,953 real) and 7,669 for testing (3,431 fake, 4,238 real).

## Methodology

1. **Text preparation** (`clean_text`): lower-casing, removing URLs, removing the *Reuters* agency tag (so the model does not learn the source instead of the content), removing punctuation and digits, stopwords (NLTK), tokens shorter than 3 letters, and lemmatisation (WordNet).
2. **Split:** 80/20, stratified, `random_state=42`. The vectoriser is fitted (`fit`) on the training set only and only applied (`transform`) to the test set, to avoid data leakage.
3. **TF-IDF:** `max_features=30000`, `ngram_range=(1, 2)`, `min_df=3`, `sublinear_tf=True`.
4. **Models:** Logistic Regression (`C=1.0`, L2), Multinomial Naive Bayes (`alpha=0.1`), Random Forest (200 trees, `max_depth=30`, `min_samples_leaf=2`).
5. **Evaluation:** hold-out on the test set (accuracy, precision, recall, F1, ROC-AUC, confusion matrix) and qualitative error analysis.

## Results (test set, 7,669 articles)

| Model | Accuracy | Precision (fake) | Recall (fake) | F1 (fake) | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | **98.40%** | 98.9% | 97.5% | 98.2% | **0.9985** |
| Random Forest | 96.49% | 98.3% | 93.7% | 96.0% | 0.9961 |
| Naive Bayes | 95.58% | 94.9% | 95.2% | 95.1% | 0.9901 |

The values are computed from the confusion matrices on the test set (`results/07_matrice_konfuzije.png`), and the table is in `results/rezultati_modela.csv`.

**Key findings**
- Logistic Regression performs best. Linear models work well on high-dimensional sparse TF-IDF matrices, and the coefficients are interpretable.
- Random Forest has high precision but misses more fake articles (215 vs 87 for Logistic Regression).
- Errors: fake articles written in a formal journalistic style pass as real, and real articles about controversial topics with emotional wording are flagged as fake. The model therefore learns **style and vocabulary**, not the truthfulness of claims.

## Limitations

- The articles come from clearly separated sources, so the classes may differ systematically in style, and the high accuracy is partly a property of the dataset. On more heterogeneous sources accuracy would probably be lower.
- English only and a narrow topic range (US politics 2015–2018).
- A single hold-out split, with no cross-validation and no systematic hyperparameter tuning.
- The lemmatiser without part-of-speech tagging makes errors (e.g. *was* becomes *wa*).
- On short examples outside the dataset the model is uncertain: for a realistic-sounding news item about an executive order (2021+) the probabilities are 50.1% fake vs 49.9% real (`predict_news()` in the notebook). The dataset covers only 2015–2018.
- Only the *Reuters* tag was removed; other formal differences between sources (e.g. a text starting with a city name) may still be a signal for the model.
- The system does not check facts; it recognises patterns in text, so it cannot replace manual verification.

## Possible improvements

The title as a separate feature, word embeddings (Word2Vec, GloVe) and fine-tuning pretrained models (e.g. BERT).

## Structure and usage

```
data/        instructions for downloading the data
notebooks/   fake_news_detection.ipynb (code with saved outputs)
results/     figures and results table
```

```bash
pip install -r requirements.txt
```

Download `Fake.csv` and `True.csv` into `data/` (see `data/README.md`), then run `jupyter notebook notebooks/fake_news_detection.ipynb`. The notebook comments are in Serbian.
