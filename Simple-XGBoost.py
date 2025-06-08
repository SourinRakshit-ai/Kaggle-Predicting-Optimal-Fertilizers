import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from google.colab import drive

# 1) Mount & cd
drive.mount('/content/drive', force_remount=True)
os.chdir('/content/drive/MyDrive/Fertilizer Prediction')

# 1a) Read train.csv
train = pd.read_csv("train.csv")

# 2) Features & target
X = train.drop(columns=["id","Fertilizer Name"])
y = train["Fertilizer Name"]

# 3) Encode target
le = LabelEncoder()
y_enc = le.fit_transform(y)

# 4) One-hot encode categoricals
X = pd.get_dummies(X, columns=["Soil Type","Crop Type"], drop_first=True)

# 5) Train/validation split
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
)

# 6) Instantiate with a fixed number of trees (no early stopping)
model = XGBClassifier(
    objective="multi:softprob",
    num_class=len(le.classes_),
    n_estimators=200,
    eval_metric="mlogloss",
    use_label_encoder=False,
    tree_method="hist",
    random_state=42
)

# 7) Fit
model.fit(
    X_tr, 
    y_tr,
    eval_set=[(X_val, y_val)],
    verbose=True
)

# 8) Evaluate
preds_val = model.predict(X_val)
acc   = accuracy_score(y_val, preds_val)
map3  = top_k_accuracy_score(y_val, model.predict_proba(X_val), k=3)
print(f"Val accuracy: {acc:.4f}, MAP@3: {map3:.4f}")

# 9) Retrain on all data & make submission
model.fit(X, y_enc)

test = pd.read_csv("test.csv")
ids  = test["id"]
X_test = pd.get_dummies(test.drop(columns="id"),
                        columns=["Soil Type","Crop Type"], drop_first=True)
X_test = X_test.reindex(columns=X.columns, fill_value=0)

proba = model.predict_proba(X_test)
top3  = np.argsort(proba, axis=1)[:, ::-1][:, :3]
names = [" ".join(le.inverse_transform(r)) for r in top3]

submission = pd.DataFrame({"id": ids, "Fertilizer Name": names})
submission.to_csv("submission_simple_xgb.csv", index=False)
print("✔ submission_simple_xgb.csv written.")
