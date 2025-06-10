import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import optuna
import xgboost as xgb
from google.colab import drive
import warnings
import logging

# ──────────────────────────────────────────────────────────────────────────────
# Setup: mount drive, create dirs, suppress warnings/logging
# ──────────────────────────────────────────────────────────────────────────────
warnings.filterwarnings('ignore', category=UserWarning)
logging.getLogger('xgboost').setLevel(logging.ERROR)

drive.mount('/content/drive', force_remount=True)
base_dir = Path('/content/drive/MyDrive/Fertilizer Prediction')
ckpt_dir = base_dir / 'checkpoints'
ckpt_dir.mkdir(parents=True, exist_ok=True)

# checkpoint paths
train_csv     = base_dir / 'train.csv'
test_csv      = base_dir / 'test.csv'
study_db      = ckpt_dir / 'optuna_map3.db'
study_pkl     = ckpt_dir / 'optuna_map3.pkl'
preproc_ckpt  = ckpt_dir / 'preprocessed.pkl'
split_ckpt    = ckpt_dir / 'split.pkl'
dtrain_buf    = ckpt_dir / 'dtrain.buffer'
dvalid_buf    = ckpt_dir / 'dvalid.buffer'
final_model_f = ckpt_dir / 'xgb_final.model'

# ──────────────────────────────────────────────────────────────────────────────
# 1) Preprocess & one-hot encode (checkpointed)
# ──────────────────────────────────────────────────────────────────────────────
if preproc_ckpt.exists():
    X, y_enc, le = joblib.load(preproc_ckpt)
else:
    df = pd.read_csv(train_csv)
    X = df.drop(columns=['id', 'Fertilizer Name'])
    le = LabelEncoder().fit(df['Fertilizer Name'])
    y_enc = le.transform(df['Fertilizer Name'])
    X = pd.get_dummies(X, columns=['Soil Type', 'Crop Type'], drop_first=True)
    joblib.dump((X, y_enc, le), preproc_ckpt)
print("✔ Preprocessed:", X.shape)

# ──────────────────────────────────────────────────────────────────────────────
# 2) Train/validation split & DMatrix (checkpointed)
# ──────────────────────────────────────────────────────────────────────────────
if split_ckpt.exists():
    X_tr, X_val, y_tr, y_val = joblib.load(split_ckpt)
else:
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y_enc,
        test_size=0.2,
        stratify=y_enc,
        random_state=42
    )
    joblib.dump((X_tr, X_val, y_tr, y_val), split_ckpt)
print("✔ Split: train", X_tr.shape, "valid", X_val.shape)

# buffer DMatrix so we can reload quickly
if not dtrain_buf.exists():
    xgb.DMatrix(X_tr, label=y_tr).save_binary(str(dtrain_buf))
if not dvalid_buf.exists():
    xgb.DMatrix(X_val, label=y_val).save_binary(str(dvalid_buf))
dtrain = xgb.DMatrix(str(dtrain_buf))
dvalid = xgb.DMatrix(str(dvalid_buf))

# ──────────────────────────────────────────────────────────────────────────────
# 3) MAP@3 metric and XGBoost feval
# ──────────────────────────────────────────────────────────────────────────────
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

def xgb_map3_eval(preds: np.ndarray, dmat: xgb.DMatrix):
    labels = dmat.get_label().astype(int)
    num_class = len(le.classes_)
    proba = preds.reshape(-1, num_class)
    return 'map@3', calculate_map_at_3(labels, proba)

# ──────────────────────────────────────────────────────────────────────────────
# 4) Optuna Bayesian tuning (checkpointed)
# ──────────────────────────────────────────────────────────────────────────────
def objective(trial: optuna.Trial) -> float:
    params = {
        "learning_rate":    trial.suggest_float("learning_rate", 1e-2, 2e-1, log=True),
        "max_depth":        trial.suggest_int("max_depth", 6, 12),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-1, 1e1, log=True),
        "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 1.0),
        "objective":        "multi:softprob",
        "num_class":        len(le.classes_),
        "tree_method":      "hist",
        "seed":             42
    }

    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        evals=[(dvalid, "validation")],
        feval=xgb_map3_eval,
        maximize=True,
        early_stopping_rounds=40,
        verbose_eval=False
    )
    proba = bst.predict(dvalid, iteration_range=(0, bst.best_iteration + 1))
    return calculate_map_at_3(y_val, proba)

storage_url = f"sqlite:///{study_db.as_posix()}"
study = optuna.create_study(
    study_name="map3_study",
    storage=storage_url,
    direction="maximize",
    load_if_exists=True
)

completed = len(study.trials)
if completed < 100:
    study.optimize(objective, n_trials=100 - completed)
    joblib.dump(study, study_pkl)

print("✔ Best MAP@3:", study.best_value)
print("✔ Best params:", study.best_params)

# ──────────────────────────────────────────────────────────────────────────────
# 5) Final model training with MAP@3 early stopping
# ──────────────────────────────────────────────────────────────────────────────
best = study.best_params.copy()
best.update({
    "objective":   "multi:softprob",
    "num_class":   len(le.classes_),
    "tree_method": "hist",
    "seed":        42
})

if final_model_f.exists():
    bst = xgb.Booster()
    bst.load_model(str(final_model_f))
else:
    bst = xgb.train(
        best,
        dtrain,
        num_boost_round=50000,
        evals=[(dvalid, "validation")],
        feval=xgb_map3_eval,
        maximize=True,
        early_stopping_rounds=100,
        verbose_eval=100
    )
    bst.save_model(str(final_model_f))

print("✔ Final model trained, best_iteration =", bst.best_iteration)

# ──────────────────────────────────────────────────────────────────────────────
# 6) Evaluate & create submission
# ──────────────────────────────────────────────────────────────────────────────
proba_val = bst.predict(dvalid, iteration_range=(0, bst.best_iteration + 1))
map3_val  = calculate_map_at_3(y_val, proba_val)
print(f"Validation MAP@3: {map3_val:.4f}")

df_test = pd.read_csv(test_csv)
X_test  = pd.get_dummies(
    df_test.drop(columns="id"),
    columns=["Soil Type", "Crop Type"],
    drop_first=True
).reindex(columns=X.columns, fill_value=0)
dtest   = xgb.DMatrix(X_test)

proba_test = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))
top3_idx   = np.argsort(proba_test, axis=1)[:, ::-1][:, :3]
names      = [" ".join(le.inverse_transform(r)) for r in top3_idx]

submission = pd.DataFrame({"id": df_test["id"], "Fertilizer Name": names})
submission.to_csv(base_dir/"submission_map3_xgb.csv", index=False)
print("✔ Submission written:", base_dir/"submission_map3_xgb.csv")
