import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, auc, confusion_matrix, precision_score, recall_score

X_train_time = np.load("data/processed/X_train_time.npy")
X_train_env = np.load("data/processed/X_train_env.npy")

X_val_time = np.load("data/processed/X_val_time.npy")
X_val_env = np.load("data/processed/X_val_env.npy")

X_test_time = np.load("data/processed/X_test_time.npy")
X_test_env = np.load("data/processed/X_test_env.npy")

y_train = np.load("data/processed/y_train.npy")
y_val = np.load("data/processed/y_val.npy")
y_test = np.load("data/processed/y_test.npy")

print("Train:", X_train_time.shape, y_train.shape)
print("Val:", X_val_time.shape, y_val.shape)
print("Test:", X_test_time.shape, y_test.shape)

# FLATTEN
X_train_time = X_train_time.reshape(X_train_time.shape[0], -1)
X_val_time = X_val_time.reshape(X_val_time.shape[0], -1)
X_test_time = X_test_time.reshape(X_test_time.shape[0], -1)

X_train = np.hstack([X_train_time, X_train_env])
X_val = np.hstack([X_val_time, X_val_env])
X_test = np.hstack([X_test_time, X_test_env])

# HANDLE IMBALANCE
pos = np.sum(y_train == 1)
neg = np.sum(y_train == 0)
scale_pos_weight = neg / (pos + 1e-6)

# MODEL
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight, 
    random_state=42
)

model.fit(X_train, y_train)

# VALIDATION THRESHOLD
val_prob = model.predict_proba(X_val)[:, 1]

best_f1, best_t = 0, 0.5

for t in np.arange(0.3, 0.7, 0.05):
    preds = (val_prob > t).astype(int)
    f1 = f1_score(y_val, preds)

    if f1 > best_f1:
        best_f1, best_t = f1, t

print("\nBest Threshold (VAL):", best_t)

# TEST
test_prob = model.predict_proba(X_test)[:, 1]
test_pred = (test_prob > best_t).astype(int)

print("\nXGBOOST")
print("Accuracy:", accuracy_score(y_test, test_pred))
print("F1:", f1_score(y_test, test_pred))
print("AUC:", roc_auc_score(y_test, test_prob))
print("Precision:", precision_score(y_test, test_pred))
print("Recall:", recall_score(y_test, test_pred))

print("\nXGBOOST")
print("Accuracy:", accuracy_score(y_test, test_pred))
print("Precision:", precision_score(y_test, test_pred))
print("Recall:", recall_score(y_test, test_pred))
print("F1:", f1_score(y_test, test_pred))
print("AUC:", roc_auc_score(y_test, test_prob))

# CONFUSION MATRIX
cm = confusion_matrix(y_test, test_pred)
tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

print("\nCONFUSION MATRIX")
print(cm)
print("TN:", tn)
print("FP:", fp)
print("FN:", fn)
print("TP:", tp)

print("\nADDITIONAL METRICS")
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)

np.save("xgb_probs.npy", test_prob)
np.save("xgb_true.npy", y_test)

print("\nSaved XGB probabilities")