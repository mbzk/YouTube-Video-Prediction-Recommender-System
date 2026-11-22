# 🎥 YouTube Video Performance Prediction & Creator Decision Support System

**A Business Intelligence Framework for YouTube Video Performance Prediction & Strategy Recommendation**

MSc Data Science & Business Analytics capstone project, Asia Pacific University (APU)
**Author:** Muhammad Bilal Zafar Khan (TP090300) | **Supervisor:** Dr. Selvakumar Samuel | **2nd Marker:** Dr. Amardeep Singh

> 🔗 **Open access repository:** https://github.com/mbzk/youtube-category-prediction
> 🚀 **Live app:** _add your Streamlit Community Cloud link here after deploying_

## Overview
Creators mostly rely on trial and error, and tools like YouTube Analytics are descriptive rather than predictive. This project classifies trending YouTube videos into **Low / Medium / High** success tiers (based on view counts) and turns the results into practical advice through a Streamlit decision-support app and a Power BI dashboard.

## Dataset
- 268,788 raw records collected from the YouTube Data API and open-access datasets, cleaned to **47,076 unique trending videos** (2020–2024, 15 categories)
- 12 engineered features: engagement rate, like/dislike/comment ratios, tag count, title length, days to trending, publish hour/day/month
- Success classes: **Low** (< 619,542 views, 30%), **Medium** (619,542 to 1,934,870, 40%), **High** (30%)
- Cleaned data: `data/youtube_final.csv`

## Models
| Model | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression | 98.11% | 0.9811 | 0.9814 | 0.9811 |
| Random Forest | 97.58% | 0.9758 | 0.9759 | 0.9758 |
| **XGBoost (final)** | **98.95%** | **0.9895** | **0.9895** | **0.9895** |

XGBoost AUC-ROC: High 0.9999, Low 0.9999, Medium 0.9996. Engagement-based features (likes, comments, ratios) are the strongest predictors; publish day and hour matter least.

## Repository structure
```
app/            Streamlit Creator Decision Support System (app.py)
models/         Trained models (XGBoost, Random Forest, Logistic Regression), scaler, label encoders
data/           Cleaned dataset and supplementary sample
notebooks/      Data collection, EDA, feature engineering and model training
dashboard/      Power BI dashboard (.pbix)
presentation/   Final capstone presentation
```

## Run locally
```bash
git clone https://github.com/mbzk/youtube-category-prediction.git
cd youtube-category-prediction
pip install -r requirements.txt
streamlit run app/app.py
```

The app has four sections: project overview, a video success predictor, an analytics dashboard, and a creator recommendation engine.

## Loading the Random Forest model
The Random Forest is stored compressed to stay under GitHub upload limits:
```python
import gzip, pickle
rf = pickle.load(gzip.open("models/rf_model.pkl.gz", "rb"))
```

## Notebook
The notebook collects data with the YouTube Data API. The API key has been removed; to rerun collection, create your own key and set it where the notebook says `YOUR_YOUTUBE_API_KEY`. Do not commit real keys.

## Power BI dashboard
`dashboard/YouTube_BI_Dashboard.pbix` needs Power BI Desktop to open (GitHub cannot preview it).

## Limitations
Survivorship bias (trending videos only), English-language/US-centred data, and a 2020–2024 period that includes COVID-era viewing anomalies.
