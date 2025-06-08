import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import optuna
import xgboost as xgb
from xgboost import XGBClassifier
from google.colab import drive
import warnings  # Import the warnings module

warnings.filterwarnings('ignore', category=UserWarning)  # Ignore all UserWarnings

import logging  # Import the logging module

logging.getLogger('xgboost').setLevel(logging.ERROR)  # Suppress XGBoost info/warnings

# 1) Mount & cd
drive.mount('/content/drive', force_remount=True)
os.chdir('/content/drive/MyDrive/Fertilizer Prediction')

# 2) Load and prepare data
df = pd.read_csv("train.csv")
X = df.drop(columns=["id", "Fertilizer Name"])
y = df["Fertilizer Name"]

# 3) Encode target
le = LabelEncoder()
y_enc = le.fit_transform(y)

# 4) One-hot encode categoricals
X = pd.get_dummies(
    X,
    columns=["Soil Type", "Crop Type"],
    drop_first=True
)

# 5) Train/validation split
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y_enc,
    test_size=0.2,
    stratify=y_enc,
    random_state=42
)

# 6) MAP@3 metric
def calculate_map_at_3(y_true: np.ndarray, proba: np.ndarray) -> float:
    ap_sum = 0.0
    n = len(y_true)
    for i, p in enumerate(proba):
        top3 = np.argsort(p)[::-1][:3]
        score = 0.0
        hits = 0
        for rank, cls in enumerate(top3):
            if cls == y_true[i]:
                hits += 1
                score += hits / (rank + 1)
        if hits > 0:
            ap_sum += score
    return ap_sum / n

# 7) Optuna objective using low-level xgb.train for early stopping
def objective(trial: optuna.Trial) -> float:
    params = {
        "learning_rate":      trial.suggest_float("learning_rate", 1e-2, 2e-1, log=True),
        "max_depth":          trial.suggest_int("max_depth", 7, 15),
        "subsample":          trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "colsample_bynode":   trial.suggest_float("colsample_bynode", 0.5, 1.0),
        "colsample_bylevel":  trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "gamma":              trial.suggest_float("gamma", 0.0, 5.0),
        "reg_lambda":         trial.suggest_float("reg_lambda", 1e-1, 1e1, log=True),
        "reg_alpha":          trial.suggest_float("reg_alpha", 0.0, 1.0),
        "objective":          "multi:softprob",
        "num_class":          len(le.classes_),
        "eval_metric":        "mlogloss",
        "tree_method":        "hist",
        "use_label_encoder":  False,
        "seed":               42
    }

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dvalid = xgb.DMatrix(X_val, label=y_val)

    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        evals=[(dvalid, "validation")],
        early_stopping_rounds=40,
        verbose_eval=False
    )

    proba = bst.predict(dvalid, iteration_range=(0, bst.best_iteration + 1))
    return calculate_map_at_3(y_val, proba)

# 8) Run Bayesian optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=40)

print("Best MAP@3 on validation:", study.best_value)
print("Best hyperparameters:")
for k, v in study.best_params.items():
    print(f"  • {k}: {v}")

# 9) Retrain final XGBClassifier on full data
best_params = study.best_params.copy()
best_params.update({
    "objective":          "multi:softprob",
    "num_class":          len(le.classes_),
    "eval_metric":        "mlogloss",
    "tree_method":        "hist",
    "use_label_encoder":  False,
    "random_state":       42,
    "n_estimators":       50000
})

final_model = XGBClassifier(**best_params)
final_model.fit(X, y_enc)

# 10) Evaluate tuned model on held-out validation
proba_val = final_model.predict_proba(X_val)
map3_val  = calculate_map_at_3(y_val, proba_val)
print(f"Tuned model MAP@3 on validation after retrain: {map3_val:.4f}")

# 11) Prepare test set & make submission
test = pd.read_csv("test.csv")
ids  = test["id"]
X_test = pd.get_dummies(
    test.drop(columns="id"),
    columns=["Soil Type", "Crop Type"],
    drop_first=True
).reindex(columns=X.columns, fill_value=0)

proba_test = final_model.predict_proba(X_test)
top3_idx   = np.argsort(proba_test, axis=1)[:, ::-1][:, :3]
names      = [" ".join(le.inverse_transform(row)) for row in top3_idx]

submission = pd.DataFrame({"id": ids, "Fertilizer Name": names})
submission.to_csv("submission_bayes_map3_xgb.csv", index=False)
print("✔ submission_bayes_map3_xgb.csv written.")
