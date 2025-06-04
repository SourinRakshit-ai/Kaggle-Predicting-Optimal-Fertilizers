# Kaggle Predicting Optimal Fertilizers

This repository contains starter code for the [Kaggle Playground Series - Season 5, Episode 6](https://www.kaggle.com/competitions/playground-series-s5e6/overview) competition.

## Getting Started

1. Download the competition data from Kaggle and place `train.csv` and `test.csv` in this directory.
2. Install dependencies:
   ```bash
   pip install pandas scikit-learn
   ```
3. Run the baseline model script:
   ```bash
   python baseline_model.py --train train.csv --test test.csv --output submission.csv
   ```
   The script prints cross‑validation accuracy and saves `submission.csv` for upload.

## Tips for Improvement

- Review feature importance from the baseline model.
- Experiment with additional algorithms such as gradient boosting or XGBoost.
- Use cross‑validation to tune hyperparameters.
- Combine multiple models for better performance.
