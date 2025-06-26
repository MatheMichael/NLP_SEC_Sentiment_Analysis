# Sentiment Analysis of 10-Q Filings from Public Companies

This project presents a simple, open-source pipeline for extracting and analyzing sentence-level sentiment from **quarterly 10-Q filings** of U.S. publicly traded companies. These filings, submitted to the **U.S. Securities and Exchange Commission (SEC)**, are legally required disclosures that are closely monitored by investors, analysts, and the financial media.

The primary output is a **Python dictionary** where:
- **Keys** are ticker symbols of selected **S&P 500 companies**
- **Values** are lists of records, each containing:
  - The **filing date**
  - The **number of positive sentences**
  - The **number of negative sentences** in the MD&A section

> **Note:** This is a **work in progress**. The sentiment data extracted here is intended for use in future analyses of how **stock prices respond to quarterly disclosures**.

---

## Workflow Overview

### 1. Selecting Companies and Downloading Filings
- A random subset of S&P 500 companies is selected, ensuring each has a full set of 10-Q filings from the past four years.
- Filings are downloaded using [`sec-edgar-downloader`](https://github.com/jadchaar/sec-edgar-downloader), which retrieves raw HTML documents from the SEC EDGAR database.

### 2. Extracting and Cleaning the MD&A Section
- The focus is on the **Management’s Discussion and Analysis (MD&A)** section — a key part of the filing where management discusses financial results, risks, and future outlook.
- Only filings that include a **table of contents** (for easier section identification) are used.
- Extracted text is cleaned to remove tables, bullet points, and excess formatting.

### 3. Sentence Segmentation
- Cleaned paragraphs are split into individual sentences using `sent_tokenize`, which is based on **Punkt**, a statistical sentence segmentation model.
- This helps preserve sentence boundaries even in the presence of abbreviations or financial formatting.

### 4. Sentiment Classification with FinBERT
- Each sentence is passed through [**FinBERT**](https://github.com/ProsusAI/finBERT), a transformer-based sentiment analysis model fine-tuned on financial text.
- Sentences are labeled as **positive**, **negative**, or **neutral**.
- Positive and negative counts are stored; neutral sentences are discarded in this version.

---

## Example Outputs from FinBERT Sentiment Classifier

```python
result = classifier("Dealers decreased inventories by $600 million during the third quarter of 2020, compared with a decrease of $300 million during the third quarter of 2021.")
print(result)
# [{'label': 'Neutral', 'score': 0.9747602343559265}]

result = classifier("Dealers increased inventories more during the nine months ended September 30, 2021, than during the nine months ended September 30, 2020. Sales increased in Asia/Pacific due to higher end-user demand for equipment and aftermarket parts, favorable currency impacts related to the Chinese yuan and Australian dollar, and the impact from changes in dealer inventories.")
print(result)
# [{'label': 'Positive', 'score': 0.9999994039535522}]

result = classifier("Financial Products’ segment profit was $173 million in the third quarter of 2021, an increase of $31 million, or 22 percent, compared with $142 million in the third quarter of 2020.")
print(result)
# [{'label': 'Positive', 'score': 0.999671459197998}]

sentence = "Sales decreased in Asia/Pacific mainly due to lower sales volume, reflecting the impact of changes in dealer inventory."
result = classifier(sentence)
print(result)
# [{'label': 'Negative', 'score': 0.999997615814209}]
```

---

## Future Directions

- Analyze the relationship between filing sentiment and **stock price movements** (e.g., earnings surprises, post-earnings drift)
- Add support for 10-K filings and other sections (e.g., Risk Factors)
- Improve MD&A section detection using NLP or rule-based fallback methods
- Visualize sentiment trends over time across sectors or companies

---

## License

This project is open-source and released under the MIT License.


