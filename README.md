# Smartphone Price Prediction Analysis

## Project Overview
This project analyzes a dataset of 1,512 smartphones to develop a predictive model for smartphone pricing. Using machine learning techniques, we explored the relationships between phone specifications, brand positioning, and market prices to create an accurate pricing model.

![Model Performance Comparison](./visualizations/model_performances.png)

## Dataset
The dataset includes various smartphone attributes:
- Technical specifications (RAM, storage, battery)
- Display features
- Video capabilities
- Brand information
- Physical characteristics
- Price information

Source: [Kaggle](https://www.kaggle.com/datasets/berkayeserr/phone-prices)

![Feature Importance](./visualizations/feature_importance.png)
*Top features driving smartphone prices*

## Key Results
Model performance comparison:

| Model | R² Score | RMSE | MAE |
|-------|----------|------|-----|
| Linear Regression | 0.925 | $75.99 | $47.69 |
| Random Forest | 0.937 | $68.84 | $34.72 |
| XGBoost | 0.961 | $51.30 | $21.80 |

![Predictions vs Actual](./visualizations/predictions_vs_actual.png)
*Model predictions compared to actual prices*


## Tools Used
- Python 3.8+
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Seaborn, Matplotlib
