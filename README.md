# Sentiment Analysis and Price Dynamics in 10-Q Filings

This project presents an open-source pipeline that extracts sentiment from quarterly 10-Q filings of U.S. publicly traded companies and analyzes how that sentiment relates to stock price behavior. 
  These filings contain forward-looking statements and management commentary that often influence investor expectations. The aim is to better understand how qualitative disclosures affect market reactions over time.

The project is divided into two main components:
1. **Sentence-level sentiment extraction** from the MD&A sections of 10-Q filings using FinBERT.
2. **Modeling and clustering the dynamics** of the relationship between sentiment and post-filing stock returns.

In the first part, we extract counts of high-confidence **positive** and **negative** sentences from each filing. From these, we compute the **Optimism Index**—defined as the ratio of positive to negative sentences—for each filing.

In the second part, we collect historical stock prices for each company around each filing date. For a given time window, we define the **return effect** as the ratio of the average post-filing price to the average pre-filing price.  

To investigate how sentiment relates to returns, we run a series of linear regressions between the **Optimism Index** and the return effect across shifting time windows. The resulting **regression slopes** measure the sensitivity of price movement to sentiment over time.
  
By varying these windows, we uncover **structured and recurring patterns** in how sentiment affects returns—some firms show early responses, others delayed or U-shaped effects. To capture these dynamics, we apply **KMeans clustering** to the time series of regression slopes, grouping firms by the shape of their sentiment–return relationship.

![Cluster 2](Figures/Cluster_figs/Cluster_2.jpg)
---

## Part 1: Extracting Sentiment from 10-Q Filings -- Workflow Overview
 
**1. Filing Collection**
- A subset of S&P 500 companies is selected, each with a consistent history of 10-Q filings over the past four years.
- Filings are retrieved via the [sec-edgar-downloader](https://github.com/jadchaar/sec-edgar-downloader) package.

**2. **Management’s Discussion and Analysis (MD&A)** Section Parsing**
- Since the 10-Q filings are raw and unstructured, only filings with a well-defined table of contents are used, which helps parsing by locating the MD&A section.
- The MD&A section is extracted and cleaned by removing tables, bullet points, and formatting artifacts.

**3. Sentence Segmentation**
- Cleaned text is split into sentences using `nltk.sent_tokenize`, which uses a statistical model to maintain robust boundary detection.

**4. Sentiment Classification with FinBERT**
- Each sentence is classified using [FinBERT](https://github.com/ProsusAI/finBERT), a transformer model fine-tuned on financial text.
- Sentences are labeled as **positive**, **negative**, or **neutral**.
- Only positive and negative of high score are counted.

### Output Format

The output of this phase is a Python dictionary where:
- **Keys** are ticker symbols
- **Values** are lists of records, each corresponding to a 10-Q filing and containing:
  - Filing date
  - Number of positive and negative sentences

These records form the input for the second part of the project, which computes the **Optimism Index**, the ratio of postive to negative counts, and links sentiment data to price behavior.

---

## Example Outputs from FinBERT

```python
result = classifier("Dealers decreased inventories by $600 million during the third quarter of 2020.")
# [{'label': 'Neutral', 'score': 0.97}]

result = classifier("Sales increased due to higher demand and favorable currency impacts.")
# [{'label': 'Positive', 'score': 0.99}]

result = classifier("Sales decreased in Asia/Pacific mainly due to lower sales volume.")
# [{'label': 'Negative', 'score': 0.99}]
```

---

## Part 2: Sentiment–Price Dynamics Analysis -- Workflow Overview

### Step 1: Price Data Collection
- For each filing, approximately 85 daily closing prices **before** and **after** the filing date are downloaded using the `yfinance` API.

### Step 2: Defining the Sentiment–Return Relationship
- An **Optimism Index** is computed for each filing as the ratio of positive to negative sentence counts.
- A series of **sliding post-filing windows** is defined to compute short-term price changes.
- In each window, the **return effect** is calculated as the ratio of average post-filing price to pre-filing price.
- Linear regressions are run using `statsmodels`, regressing the normalized optimism index on the normalized return effect.
- The slope coefficient reflects the **sensitivity of stock price behavior to sentiment** at each window position.

### Step 3: Time-Series Pattern Analysis

Each ticker yields a time series of slope values over sliding windows.  These trajectories capture the evolving relationship between filing sentiment and market response. Interestingly, the plots show structure and similarities across stocks, motivating classification.

---

## Randomization Check: Validating Patterns

To ensure the observed patterns are not artifacts, a **permutation test** is performed:
- Sentiment indices are randomly permuted for each ticker, and the slope estimation is repeated.
- Resulting patterns appear random, suggesting that the original relationships are statistically meaningful.

---

## Clustering Sentiment–Return Dynamics

To group firms with similar sentiment–price dynamics:
- **KMeans clustering** is applied to the standardized slope sequences and their first derivatives.
- This captures firms with similar **temporal profiles** of sentiment impact, such as:
  - Immediate positive/negative effects
  - Delayed market responses
  - Reversal or U-shaped patterns

---

## Summary of Findings

- The Optimism Index, derived from sentence-level sentiment, exhibits **meaningful correlations** with returns.
- **Slope trajectory clustering** reveals distinct behavioral patterns in how different firms’ sentiment influences stock price movements.
- Standardization allows shape-based clustering, though **absolute magnitude and direction** of impact require further modeling.

---

## Future Work

- Scale the dataset to include more firms and filing periods.
- Train a **neural network model** (e.g., 1D CNN or autoencoder) to classify slope shapes.
- Extend analysis to additional filing types (e.g., 10-K) and other text sections (e.g., Risk Factors).

---
## Repository Contents

- `Fetching_Sentiments_10Q-checkpoint.ipynb`  
  Notebook for downloading 10-Q filings and extracting sentence-level sentiment.

- `Sentiment_Analysis.ipynb`  
  Notebook for analyzing sentiment dynamics and their relationship to stock price behavior.
- `figures/` folder:
    - `cluster_figures/`  
       Folder containing visualizations of clustered slope patterns from the sentiment–price dynamics analysis.

    - `optimism_index_plot.png`  
       Plot showing the evolution of the optimism index over time for each ticker.
---
## License

This project is open-source and released under the MIT License.
