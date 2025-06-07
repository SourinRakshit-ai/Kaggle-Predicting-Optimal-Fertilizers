# ──────────────────────────────────────────────────────────────────────────────
# Block 1: Imports, Paths & GPU Check
# ──────────────────────────────────────────────────────────────────────────────
import os, time, joblib
from pathlib import Path

# Ensure checkpoint directories exist
base_dir    = Path("checkpoints");      base_dir.mkdir(exist_ok=True)
optuna_dir  = base_dir / "optuna";      optuna_dir.mkdir(exist_ok=True)
models_dir  = base_dir / "models";      models_dir.mkdir(exist_ok=True)
data_dir    = Path("/kaggle/input/playground-series-s5e6")
train_csv   = data_dir / "train.csv"
test_csv    = data_dir / "test.csv"
sample_csv  = data_dir / "sample_submission.csv"

# GPU check
import subprocess
gpu_info = subprocess.run(
    ["nvidia-smi","--query-gpu=name,compute_cap","--format=csv,noheader"],
    stdout=subprocess.PIPE, text=True
).stdout.strip()
if not gpu_info:
    raise RuntimeError("No GPU found – switch to a GPU runtime.")
print("GPU detected:", gpu_info)

# Common imports
import numpy as np, pandas as pd
from tqdm import tqdm
from sklearn.model_selection   import StratifiedKFold, train_test_split
from sklearn.preprocessing    import LabelEncoder, PolynomialFeatures, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model     import LogisticRegression
from category_encoders        import CatBoostEncoder
import xgboost as xgb
import lightgbm as lgb
import optuna
from optuna.exceptions        import TrialPruned
from xgboost                   import callback as xgb_callback


# ──────────────────────────────────────────────────────────────────────────────
# Block 2: MAP@3 & Pruning Callback
# ──────────────────────────────────────────────────────────────────────────────
def calculate_map_at_3(y_true, y_pred_proba):
    ap_sum = 0.0
    y_true = np.array(y_true)
    for i, probs in enumerate(y_pred_proba):
        top3 = np.argsort(probs)[::-1][:3]
        hits = [(1.0/(k+1.0)) for k,p in enumerate(top3) if p==y_true[i]]
        if hits: ap_sum += hits[0]
    return ap_sum/len(y_true)

class MAP3PruningCallback(xgb_callback.TrainingCallback):
    def __init__(self, trial, dval, y_val):
        self.trial, self.dval, self.y_val = trial, dval, y_val
    def after_iteration(self, model, epoch, evals_log):
        proba = model.predict(self.dval, iteration_range=(0, epoch+1))
        score = calculate_map_at_3(self.y_val, proba)
        self.trial.report(score, step=epoch)
        if self.trial.should_prune():
            raise TrialPruned()
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Block 3: Raw Preprocessing (Ratios + Soil–Crop Embeddings + Polynomial)
# ──────────────────────────────────────────────────────────────────────────────
raw_ckpt = base_dir / "raw_preprocessed.pkl"

if raw_ckpt.exists():
    # Load cached preprocessing
    X_num_poly, y, weights, le, soil_le, crop_le, poly, num_cols = joblib.load(raw_ckpt)
    print("✅ Loaded raw preprocessing:", X_num_poly.shape)
else:
    # 3.1) Read data & label‐encode target
    df     = pd.read_csv(train_csv)
    y_raw  = df["Fertilizer Name"].values
    le     = LabelEncoder().fit(y_raw)
    y      = le.transform(y_raw)

    # 3.2) Compute nutrient-ratio features
    df_num = df[["Temparature","Humidity","Moisture","Nitrogen","Potassium","Phosphorous"]].copy()
    eps    = 1e-6
    N, P, K = df_num["Nitrogen"], df_num["Phosphorous"], df_num["Potassium"]
    df_num["N_over_P"]  = N/(P+eps)
    df_num["N_over_K"]  = N/(K+eps)
    df_num["P_over_K"]  = P/(K+eps)
    df_num["total_NPK"] = N+P+K
    df_num["diff_N_P"]  = N-P
    df_num["diff_P_K"]  = P-K

    # 3.3) Prepare soil & crop IDs for embeddings
    soil_le = LabelEncoder().fit(df["Soil Type"])
    crop_le = LabelEncoder().fit(df["Crop Type"])
    s_ids   = soil_le.transform(df["Soil Type"]).astype("int32").reshape(-1,1)
    c_ids   = crop_le.transform(df["Crop Type"]).astype("int32").reshape(-1,1)
    n_soil, n_crop = len(soil_le.classes_), len(crop_le.classes_)

    # 3.4) Build & train tiny Keras embedding model
    import tensorflow as tf
    from tensorflow.keras import Input, Model
    from tensorflow.keras.layers import Embedding, Flatten, Concatenate, Dense, Dropout
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import EarlyStopping
    tf.random.set_seed(42)

    # One-hot encode target and cast to float32
    y_cat = to_categorical(y, num_classes=len(le.classes_)).astype("float32")

    # Define embedding network
    soil_input = Input(shape=(1,), name="soil_input")
    crop_input = Input(shape=(1,), name="crop_input")
    soil_emb   = Flatten()(Embedding(input_dim=n_soil, output_dim=8, name="soil_emb")(soil_input))
    crop_emb   = Flatten()(Embedding(input_dim=n_crop, output_dim=8, name="crop_emb")(crop_input))
    x = Concatenate()([soil_emb, crop_emb])
    x = Dense(32, activation="relu")(x); x = Dropout(0.2)(x)
    x = Dense(16, activation="relu")(x); x = Dropout(0.2)(x)
    output = Dense(len(le.classes_), activation="softmax")(x)

    emb_model = Model(inputs=[soil_input, crop_input], outputs=output)
    emb_model.compile(optimizer="adam",
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])

    # 3.4a) Train on 90/10 split using list inputs
    from sklearn.model_selection import train_test_split
    s_tr, s_val, c_tr, c_val, ytr, yval = train_test_split(
        s_ids, c_ids, y_cat,
        test_size=0.10,
        stratify=y,
        random_state=42
    )

    emb_model.fit(
        [s_tr, c_tr], ytr,
        validation_data=([s_val, c_val], yval),
        epochs=20,
        batch_size=1024,
        callbacks=[EarlyStopping("val_loss", patience=3, restore_best_weights=True)],
        verbose=2
    )

    # 3.4b) Extract embedding weight matrices
    soil_emb_weights = emb_model.get_layer("soil_emb").get_weights()[0]  # shape (n_soil,8)
    crop_emb_weights = emb_model.get_layer("crop_emb").get_weights()[0]  # shape (n_crop,8)

    # 3.5) Map embeddings back to each row
    soil_feats = soil_emb_weights[s_ids.flatten()]
    crop_feats = crop_emb_weights[c_ids.flatten()]

    # 3.6) Concatenate numeric+ratio + embeddings
    df_concat = pd.concat(
        [df_num.reset_index(drop=True),
         pd.DataFrame(soil_feats, columns=[f"soil_emb_{i}" for i in range(8)]),
         pd.DataFrame(crop_feats, columns=[f"crop_emb_{i}" for i in range(8)])],
        axis=1
    )

    # 3.7) Polynomial features on all these
    num_cols = df_concat.columns.tolist()
    poly     = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly   = poly.fit_transform(df_concat.values)
    poly_cols = poly.get_feature_names_out(num_cols)
    X_num_poly = pd.DataFrame(X_poly, columns=poly_cols)
    print("✅ Built polynomial features:", X_num_poly.shape)

    # 3.8) Compute class weights
    from sklearn.utils.class_weight import compute_class_weight
    cw = compute_class_weight("balanced", classes=np.unique(y), y=y)
    weights = cw[y]

    # 3.9) Checkpoint everything needed downstream
    joblib.dump((X_num_poly, y, weights, le, soil_le, crop_le, poly, num_cols),
                raw_ckpt)
    print("✅ Saved raw preprocessing to", raw_ckpt)


# ──────────────────────────────────────────────────────────────────────────────
# Block 4: Create Stratified Subset for Optuna (with tqdm)
# ──────────────────────────────────────────────────────────────────────────────
sample_ckpt = base_dir/"sampled.pkl"
if sample_ckpt.exists():
    X_s, y_s, w_s = joblib.load(sample_ckpt)
    print("✅ Loaded subset:", X_s.shape)
else:
    print("▶ Sampling 200k rows stratified by class…")
    idxs = []
    rng = np.random.RandomState(42)
    total = len(y)
    target_size = min(200_000, total)
    for cls in np.unique(y):
        cls_idx = np.where(y==cls)[0]
        keep = int(round(target_size * len(cls_idx)/total))
        idxs.append(rng.choice(cls_idx, keep, replace=False))
    idx = np.concatenate(idxs)
    if len(idx)>target_size:
        idx = rng.choice(idx, target_size, replace=False)
    X_s, y_s, w_s = X_num_poly.iloc[idx], y[idx], weights[idx]
    joblib.dump((X_s,y_s,w_s), sample_ckpt)
    print("✅ Saved subset:", X_s.shape)


# ──────────────────────────────────────────────────────────────────────────────
# Block 5: Fast Optuna Hyperparameter Tuning (XGB + LGB on GPU)
# ──────────────────────────────────────────────────────────────────────────────

from optuna.exceptions import TrialPruned
from lightgbm import Dataset as LGBDataset
import optuna
import xgboost as xgb
from xgboost import callback as xgb_callback
from sklearn.model_selection import StratifiedKFold

def tune_xgb_fast(train_X, train_y, train_w, n_trials=30):
    study_file = optuna_dir/"xgb_dart_fast.db"
    study = optuna.create_study(
        study_name="xgb_dart_fast",
        storage=f"sqlite:///{study_file}",
        direction="maximize",
        load_if_exists=True
    )

    def objective(trial):
        params = {
            "objective":       "multi:softprob",
            "num_class":       len(le.classes_),
            "tree_method":     "gpu_hist",
            "predictor":       "gpu_predictor",
            "device":          "cuda:0",
            "booster":         "dart",
            "learning_rate":   trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
            "max_leaves":      trial.suggest_int("max_leaves", 64, 128),
            "min_child_weight":trial.suggest_float("min_child_weight", 1e-3, 10, log=True),
            "subsample":       trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda":      trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
            "eval_metric":     "mlogloss",
            "verbosity":       0,
        }

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []

        for fold, (ti, vi) in enumerate(skf.split(train_X, train_y)):
            dtr = xgb.DMatrix(train_X.iloc[ti], train_y[ti], weight=train_w[ti])
            dvl = xgb.DMatrix(train_X.iloc[vi], train_y[vi], weight=train_w[vi])

            callbacks = [ xgb_callback.EarlyStopping(rounds=10) ]
            if fold == 0:
                callbacks.append(MAP3PruningCallback(trial, dvl, train_y[vi]))

            bst = xgb.train(
                params,
                dtr,
                num_boost_round=1000,
                evals=[(dvl, "val")],
                callbacks=callbacks,
                verbose_eval=False
            )

            preds = bst.predict(dvl, iteration_range=(0, bst.best_iteration + 1))
            scores.append(calculate_map_at_3(train_y[vi], preds))

        return float(np.mean(scores))

    study.optimize(
        objective,
        n_trials=n_trials,
        catch=(TrialPruned,),
        show_progress_bar=True
    )
    return study.best_trial.params


def tune_lgb_fast(train_X, train_y, train_w, n_trials=15):
    study_file = optuna_dir/"lgb_fast.db"
    study = optuna.create_study(
        study_name="lgb_fast",
        storage=f"sqlite:///{study_file}",
        direction="maximize",
        load_if_exists=True
    )

    def objective(trial):
        params = {
            "objective":        "multiclass",
            "num_class":        len(le.classes_),
            "metric":           "multi_logloss",
            "device":           "gpu",
            "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
            "num_leaves":       trial.suggest_int("num_leaves", 64, 128),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 200),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "verbosity":       -1,
        }

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []

        for ti, vi in skf.split(train_X, train_y):
            dtr = LGBDataset(train_X.iloc[ti], train_y[ti], weight=train_w[ti])
            dvl = LGBDataset(train_X.iloc[vi], train_y[vi], weight=train_w[vi])

            gbm = lgb.train(
                params,
                dtr,
                valid_sets=[dvl],
                num_boost_round=1000,
                early_stopping_rounds=10,
                verbose=False
            )

            preds = gbm.predict(train_X.iloc[vi], num_iteration=gbm.best_iteration)
            scores.append(calculate_map_at_3(train_y[vi], preds))

        return float(np.mean(scores))

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True
    )
    return study.best_trial.params


# Execute the faster tuning
xgb_fast_best = tune_xgb_fast(X_s, y_s, w_s, n_trials=20)
#lgb_fast_best = tune_lgb_fast(X_s, y_s, w_s, n_trials=15)
print("✅ Fast XGB params:", xgb_fast_best)
#print("✅ Fast LGB params:", lgb_fast_best)


# ──────────────────────────────────────────────────────────────────────────────
# Block 6: Full‐Data DMatrix & Final Model Training (Bagged XGB + LGB)
# ──────────────────────────────────────────────────────────────────────────────
# 6.1 Train/hold split
split_ckpt = base_dir/"full_split.pkl"
if split_ckpt.exists():
    X_tr, y_tr, w_tr, X_hd, y_hd, w_hd = joblib.load(split_ckpt)
else:
    X_tr, X_hd, y_tr, y_hd, w_tr, w_hd = train_test_split(
      X_num_poly.values, y, weights, test_size=0.2,
      stratify=y, random_state=42
    )
    joblib.dump((X_tr,y_tr,w_tr,X_hd,y_hd,w_hd), split_ckpt)

# 6.2 Build binary DMatrix and save
dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
dtrain.save_binary(models_dir/"dtrain.buffer")
dtrain = xgb.DMatrix(str(models_dir/"dtrain.buffer"))

dhold = xgb.DMatrix(X_hd, label=y_hd, weight=w_hd)
dhold.save_binary(models_dir/"dhold.buffer")
dhold = xgb.DMatrix(str(models_dir/"dhold.buffer"))

# 6.3 Bagged XGB (seeds)
seeds=[0,42,2025,1234]
xgb_models=[]
for s in seeds:
    fn=models_dir/f"xgb_dart_{s}.bin"
    if fn.exists():
        xgb_models.append(joblib.load(fn))
    else:
        params = {
            **xgb_best,
            "objective":"multi:softprob","num_class":len(le.classes_),
            "tree_method":"gpu_hist","gpu_id":0,"verbosity":0
        }
        params.update({"random_state":s})
        bst=xgb.train(params,dtrain,2000,evals=[(dhold,"val")],
                      early_stopping_rounds=50,verbose_eval=False)
        joblib.dump(bst,fn); xgb_models.append(bst)

# 6.4 Final LightGBM on full data
lgb_fn = models_dir/"lgb_full.bin"
if lgb_fn.exists():
    lgb_model = joblib.load(lgb_fn)
else:
    lgb_train = lgb.Dataset(X_tr,label=y_tr,weight=w_tr)
    params = {"device":"gpu", **lgb_best, "objective":"multiclass","num_class":len(le.classes_)}
    lgb_model = lgb.train(params,lgb_train,2000,valid_sets=[lgb_train],
                          early_stopping_rounds=50,verbose=False)
    joblib.dump(lgb_model,lgb_fn)

print("✅ Trained bagged XGB and full-data LGB")
