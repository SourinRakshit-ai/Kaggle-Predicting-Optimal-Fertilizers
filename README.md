 # Predicting Optimal Fertilizer with XGBoost & MAP\@3 Optimization

**This repository contains a solution for the Kaggle Playground Series S5E6 competition: “Fertilizer Prediction.”**
The goal is to recommend the top 3 fertilizers for a given field, based on environmental readings and soil/crop types, and to maximize the Mean Average Precision @ 3 (MAP\@3) metric.

---

## 📋 Overview

* **Problem**
  Given samples of Temperature, Humidity, Moisture, N/P/K measurements plus Soil Type and Crop Type, predict the best three fertilizers (ranked) for each test sample.

* **Data**

  * `train.csv`: 750 000 labeled examples (`id`, features, `Fertilizer Name`).
  * `test.csv`: 250 000 unlabeled examples (same features, no label).
  * Metric: **MAP\@3** (a correct fertilizer in the top 3 predictions, weighted by rank).

* **Our Approach**

  1. **Preprocessing & Feature Engineering**

     * One-hot encode categorical features.
     * (Optional) ratio and polynomial features to capture non-linear interactions.
  2. **Modeling**

     * **XGBoost** (`multi:softprob`) as the base learner.
     * **Bayesian hyperparameter tuning** (Optuna) directly optimizing MAP\@3 via a custom evaluation function.
     * **Early stopping** on MAP\@3 to avoid overfitting.
     * **Calibration** of predicted probabilities (Platt scaling) on a held-out set.
  3. **Submission**

     * Retrain final booster on full data with best params.
     * Predict test probabilities → take top 3 classes → format into `submission.csv`.

---

## 🚀 Quick Start

1. **Clone this repo**

   ```bash
   git clone https://github.com/your-username/fertilizer-prediction.git
   cd fertilizer-prediction
   ```

2. **Install requirements**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download data**

   * Place `train.csv` and `test.csv` from the [Kaggle Playground S5E6 competition](https://www.kaggle.com/competitions/playground-series-s5e6/overview) into `data/`.

4. **Run the pipeline**

   ```bash
   python run_pipeline.py
   ```

   This will automatically:

   * Preprocess and checkpoint features.
   * Perform Bayesian optimization (100 Optuna trials) maximizing MAP\@3.
   * Train the final XGBoost model with MAP\@3 early stopping.
   * Calibrate and generate `submission_map3_xgb.csv`.

---

## 📈 Results

* **Final validation MAP\@3:** `0.3422`
* **Public leaderboard score:** *coming soon*

---

## 🔧 Customization

* **Feature Engineering**
  Tweak the preprocessing in `preprocess.py`: add ratio, polynomial, or embedding features.
* **Hyperparameter Ranges**
  Modify `objective()` in `optuna_tune.py` to broaden or narrow search intervals.
* **Model Choice**
  Switch to LightGBM or CatBoost by replacing the `xgb.train` calls with their respective APIs (with a custom MAP\@3 feval).

---

## 📜 License

This work is released under the [MIT License](LICENSE). See `LICENSE` for details.

---

> **Note:** This is a proof-of-concept for learning purposes. For production deployment, further validation and robustness checks are recommended.
