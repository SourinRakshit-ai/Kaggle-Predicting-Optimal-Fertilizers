## Importing the Libraries and Loading the dataset
import os
import time
import joblib
import pandas as pd
import numpy as np
import category_encoders as ce
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from xgboost import callback as xgb_callback
import optuna
from optuna.integration import XGBoostPruningCallback

# If running in Colab:
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
os.chdir('/content/drive/MyDrive/Fertilizer Prediction')
print("Working directory:", os.getcwd())

## Defining MAP@3 metric
def calculate_map_at_3(y_true, y_pred_proba):
    """
    Compute Mean Average Precision @3 for multiclass predictions.
    """
    ap_sum = 0.0
    num_samples = len(y_true)
    y_true = np.array(y_true)

    for i in range(num_samples):
        actual = y_true[i]
        top3 = np.argsort(y_pred_proba[i])[::-1][:3]
        score = 0.0
        num_hits = 0.0
        for rank, pred_class in enumerate(top3):
            if pred_class == actual:
                num_hits += 1.0
                score += num_hits / (rank + 1.0)
        if num_hits > 0:
            ap_sum += score / 1.0

    return ap_sum / num_samples

## Step 1: Load & Preprocess

preprocessed_path = "X_processed_df_cpu.pkl"
encoder_path      = "preprocessing_objects_cpu.pkl"  # will store (le, te, poly, numeric_cols, cat_cols, poly_feature_names)

if os.path.exists(preprocessed_path) and os.path.exists(encoder_path):
    print("▶ Loading preprocessed features and preprocessing objects from disk...")
    X_processed_df, y, weights = joblib.load(preprocessed_path)
    le, te, poly, numeric_cols, categorical_cols, poly_feature_names = joblib.load(encoder_path)
    print(f"    ✔ Loaded: X_processed_df.shape = {X_processed_df.shape}")
else:
    print("▶ Preprocessing raw data (this may take a moment)…")
    # 1a) Load raw CSV
    train_df = pd.read_csv("train.csv")
    X_raw    = train_df.drop(columns=["id", "Fertilizer Name"])
    y_raw    = train_df["Fertilizer Name"]

    # 1b) Label-encode the target
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    print(f"    ✔ Found {len(le.classes_)} fertilizer classes.")

    # 1c) Identify numeric vs. categorical columns
    numeric_cols     = X_raw.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X_raw.select_dtypes(exclude=np.number).columns.tolist()
    print(f"    Numeric cols: {numeric_cols}")
    print(f"    Categorical cols: {categorical_cols}")

    # 1d) Target-encode categorical columns (global; note: this leaks labels into features)
    te = ce.TargetEncoder(cols=categorical_cols)
    X_cat_te = te.fit_transform(X_raw[categorical_cols], y)

    # 1e) Polynomial interaction features for numeric columns
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_num_poly = poly.fit_transform(X_raw[numeric_cols])
    poly_feature_names = poly.get_feature_names_out(numeric_cols).tolist()

    # 1f) Combine numeric + poly + target-encoded cat into one DataFrame
    X_numeric = X_raw[numeric_cols].reset_index(drop=True)
    X_poly    = pd.DataFrame(X_num_poly, columns=poly_feature_names)
    X_cat_te  = X_cat_te.reset_index(drop=True)

    X_processed_df = pd.concat([X_numeric, X_poly, X_cat_te], axis=1)
    print(f"    ✔ Processed features shape: {X_processed_df.shape}")

    # 1g) Compute class weights
    class_weights_array = compute_class_weight("balanced", classes=np.unique(y), y=y)
    weights = np.array([class_weights_array[label] for label in y])
    print(f"    ✔ Computed class weights.")

    # 1h) Save preprocessed result + all encoder objects
    joblib.dump((X_processed_df, y, weights), preprocessed_path)
    joblib.dump((le, te, poly, numeric_cols, categorical_cols, poly_feature_names), encoder_path)
    print("    ✔ Saved preprocessing to disk.")


## Step 2: SAMPLE A STRATIFIED SUBSET FOR OPTUNA TUNING  

sampled_path = "X_sampled_cpu.pkl"
if not os.path.exists(sampled_path):
    print("▶ Creating stratified subset (200k rows max) for Optuna tuning…")
    sample_size = 200_000
    rng = np.random.RandomState(42)

    # Ensure sample_size ≤ total rows
    n_total = len(y)
    if sample_size > n_total:
        sample_size = n_total

    idx_per_class = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        n_cls   = len(cls_idx)

        # proportional number for this class
        n_keep = int(np.round(sample_size * (n_cls / n_total)))
        n_keep = min(n_keep, n_cls)  # clamp to available samples

        chosen = rng.choice(cls_idx, size=n_keep, replace=False)
        idx_per_class.append(chosen)

    sample_idx = np.concatenate(idx_per_class)
    # Adjust if rounding gave too few/many
    if len(sample_idx) > sample_size:
        sample_idx = rng.choice(sample_idx, size=sample_size, replace=False)
    elif len(sample_idx) < sample_size:
        remaining = np.setdiff1d(np.arange(n_total), sample_idx)
        extra = rng.choice(remaining, size=(sample_size - len(sample_idx)), replace=False)
        sample_idx = np.concatenate([sample_idx, extra])

    # Subset data
    X_sample = X_processed_df.iloc[sample_idx].reset_index(drop=True)
    y_sample = y[sample_idx]
    w_sample = weights[sample_idx]

    joblib.dump((X_sample, y_sample, w_sample), sampled_path)
    print(f"    ✔ Saved sampled subset: {X_sample.shape[0]} rows.")
else:
    print("▶ Loading existing sampled subset from disk…")
    X_sample, y_sample, w_sample = joblib.load(sampled_path)
    print(f"    ✔ Loaded sampled subset: {X_sample.shape[0]} rows.")


## Step 3: Define Custom MAP@3 Pruning Callback

class MAP3PruningCallback(xgb_callback.TrainingCallback):
    """
    XGBoost callback that computes MAP@3 on the validation DMatrix each iteration,
    reports it to Optuna, and prunes if needed.
    """
    def __init__(self, trial, dval, y_val):
        self.trial = trial
        self.dval  = dval
        self.y_val = y_val

    def after_iteration(self, model, epoch, evals_log):
        proba     = model.predict(self.dval, iteration_range=(0, epoch + 1))
        map3_score = calculate_map_at_3(self.y_val, proba)
        self.trial.report(map3_score, step=epoch)
        if self.trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        return False

## Step 4: Optuna Tuning (5-Fold CV on Sample, MAP@3 Pruning)

study_name   = "xgb_cpu_5fold_map3"
storage_name = f"sqlite:///{study_name}.db"
study_path   = f"{study_name}_study.pkl"
N_TRIALS     = 40

def objective(trial):
    params = {
        "objective":  "multi:softprob",
        "num_class":  len(le.classes_),
        "tree_method": "hist",
        "n_jobs":     -1,
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
        "max_depth":      trial.suggest_int("max_depth", 3, 20),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-5, 10.0, log=True),
        "gamma":          trial.suggest_float("gamma", 0.0, 5.0),
        "subsample":      trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha":      trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda":     trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "max_bin":        trial.suggest_int("max_bin", 256, 2048),
        "grow_policy":    trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
        "verbosity":      0,
        "eval_metric":    "mlogloss"
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    map3_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_sample, y_sample)):
        X_tr = X_sample.iloc[train_idx].values
        y_tr = y_sample[train_idx]
        w_tr = w_sample[train_idx]

        X_va = X_sample.iloc[val_idx].values
        y_va = y_sample[val_idx]
        w_va = w_sample[val_idx]

        dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
        dval   = xgb.DMatrix(X_va, label=y_va, weight=w_va)

        callbacks = [
            xgb_callback.EarlyStopping(rounds=50, save_best=True)
        ]
        if fold_idx == 0:
            callbacks.append(MAP3PruningCallback(trial, dval, y_va))

        bst = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=2000,
            evals=[(dval, "validation")],
            callbacks=callbacks,
            verbose_eval=False
        )

        proba_va = bst.predict(dval, iteration_range=(0, bst.best_iteration + 1))
        map3_scores.append(calculate_map_at_3(y_va, proba_va))

    return float(np.mean(map3_scores))


# 4a) Create or resume the study
if os.path.exists(f"{study_name}.db"):
    print("▶ Found existing Optuna DB; resuming study…")
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        load_if_exists=True
    )
else:
    print("▶ Creating new Optuna study (5-Fold CPU MAP@3)…")
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        load_if_exists=True
    )

# 4b) Run only the remaining trials
completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
if completed_trials < N_TRIALS:
    to_go = N_TRIALS - completed_trials
    print(f"▶ Completed {completed_trials}/{N_TRIALS} trials. Running {to_go} more…")
    try:
        study.optimize(objective, n_trials=to_go, catch=(optuna.exceptions.TrialPruned,))
    except KeyboardInterrupt:
        print("▶ Optimization interrupted by user; partial results saved.")
else:
    print(f"▶ Already completed {completed_trials}/{N_TRIALS} trials; skipping optimization.")

# 4c) Save updated study and print best trial
joblib.dump(study, study_path)
completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
print(f"    ✔ Saved Optuna study to '{study_path}' ({completed_trials}/{N_TRIALS} done).")

best_trial = study.best_trial
print("=== Best Trial so far ===")
print(f"  MAP@3 = {best_trial.value:.6f}")
for key, val in best_trial.params.items():
    print(f"    • {key}: {val}")


## Step 5: Split Full Data Into Train/Hold-out & Save Arrays


arrays_path = "full_split_arrays_cpu.pkl"
if os.path.exists(arrays_path):
    print("▶ Loading existing train/hold-out arrays from disk…")
    X_tr_full, y_tr_full, w_tr_full, X_hold, y_hold, w_hold = joblib.load(arrays_path)
else:
    print("▶ Creating train/hold-out split arrays…")
    X_full = X_processed_df.values
    y_full = y
    w_full = weights

    X_tr_full, X_hold, y_tr_full, y_hold, w_tr_full, w_hold = train_test_split(
        X_full, y_full, w_full,
        test_size=0.20, random_state=42, stratify=y_full
    )

    joblib.dump((X_tr_full, y_tr_full, w_tr_full, X_hold, y_hold, w_hold), arrays_path)
    print(f"    ✔ Saved raw train/hold-out arrays to '{arrays_path}'")


## Step 6: Train or Load Ensemble Boosters

ensemble_seeds = [0, 42, 50, 2025] 
booster_folder  = "boosters_cpu"
os.makedirs(booster_folder, exist_ok=True)

# Prepare full DMatrix objects
dtrain_full = xgb.DMatrix(X_tr_full, label=y_tr_full, weight=w_tr_full)
dhold       = xgb.DMatrix(X_hold,    label=y_hold,    weight=w_hold)

# 6a) Build final_params by merging best_trial.params
final_params = {
    "objective":   "multi:softprob",
    "num_class":   len(le.classes_),
    "tree_method": "hist",
    "n_jobs":      -1,
    "eval_metric": "mlogloss",
    "verbosity":   1,
    **best_trial.params
}

# 6b) Train or load each booster (seeded for an ensemble)
boosters = []
for seed in ensemble_seeds:
    filename = os.path.join(booster_folder, f"best_xgb_cpu_seed{seed}.dill")
    if os.path.exists(filename):
        print(f"▶ Loading existing booster: {filename}")
        bst = joblib.load(filename)
    else:
        print(f"▶ Training booster with seed={seed} …")
        params = final_params.copy()
        params["random_state"] = seed

        bst = xgb.train(
            params=params,
            dtrain=dtrain_full,
            num_boost_round=3000,
            evals=[(dhold, "validation")],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        joblib.dump(bst, filename)
        print(f"    ✔ Saved booster to '{filename}'")
    boosters.append(bst)
print("▶ All boosters ready.")
