import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, auc, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt

from model import MultimodalModel

# REPRODUCIBILITY
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)

# DEVICE
device = torch.device("cpu")
print("Using device:", device)

# HYPERPARAMETER
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0
EPOCHS = 30
PATIENCE = 5

# DATA
X_train_time = np.load("data/processed/X_train_time.npy")
X_train_env = np.load("data/processed/X_train_env.npy")
X_val_time = np.load("data/processed/X_val_time.npy")
X_val_env = np.load("data/processed/X_val_env.npy")
X_test_time = np.load("data/processed/X_test_time.npy")
X_test_env = np.load("data/processed/X_test_env.npy")

y_train = np.load("data/processed/y_train.npy")
y_val = np.load("data/processed/y_val.npy")
y_test = np.load("data/processed/y_test.npy")

# TO TENSOR
to_tensor = lambda x: torch.tensor(x, dtype=torch.float32)

X_train_time = to_tensor(X_train_time)
X_train_env = to_tensor(X_train_env)
y_train = to_tensor(y_train)

X_val_time = to_tensor(X_val_time)
X_val_env = to_tensor(X_val_env)
y_val = to_tensor(y_val)

X_test_time = to_tensor(X_test_time)
X_test_env = to_tensor(X_test_env)
y_test = to_tensor(y_test)

# DATASET
train_dataset = TensorDataset(X_train_time, X_train_env, y_train)
val_dataset = TensorDataset(X_val_time, X_val_env, y_val)
test_dataset = TensorDataset(X_test_time, X_test_env, y_test)

# SAMPLER
y_np = y_train.numpy().astype(int)
weights = 1.0 / (np.bincount(y_np) + 1e-6)
sample_weights = weights[y_np]

sampler = WeightedRandomSampler(sample_weights, len(sample_weights), True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# MODEL
model = MultimodalModel(
    X_train_time.shape[2],
    X_train_env.shape[1]
).to(device)

# LOSS
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
pos_weight = pos_weight.clone().detach().float().to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

# TRAIN
best_val_loss = float("inf")
patience = PATIENCE
counter = 0

train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for Xt, Xe, yb in train_loader:
        Xt, Xe, yb = Xt.to(device), Xe.to(device), yb.to(device)

        optimizer.zero_grad()
        out = model(Xt, Xe).squeeze()
        loss = criterion(out, yb)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # VALIDATION
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for Xt, Xe, yb in val_loader:
            Xt, Xe, yb = Xt.to(device), Xe.to(device), yb.to(device)
            out = model(Xt, Xe).squeeze()
            val_loss += criterion(out, yb).item()

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1} | Train {train_loss:.2f} | Val {val_loss:.2f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        counter += 1

    if counter >= patience:
        print("Early stopping!")
        break

# LOAD BEST MODEL
model.load_state_dict(torch.load("best_model.pth", weights_only=True))

# TRAIN AUC
model.eval()
train_probs, train_true = [], []

with torch.no_grad():
    for Xt, Xe, yb in train_loader:
        Xt, Xe = Xt.to(device), Xe.to(device)
        p = torch.sigmoid(model(Xt, Xe).squeeze())

        train_probs.extend(p.cpu().numpy())
        train_true.extend(yb.numpy())

train_probs = np.array(train_probs)
train_true = np.array(train_true)

print("\ngit add .TRAIN PERFORMANCE")
print("Train AUC:", roc_auc_score(train_true, train_probs))

# TEST
probs, true = [], []

with torch.no_grad():
    for Xt, Xe, yb in test_loader:
        Xt, Xe = Xt.to(device), Xe.to(device)
        p = torch.sigmoid(model(Xt, Xe).squeeze())

        probs.extend(p.cpu().numpy())
        true.extend(yb.numpy())

probs = np.array(probs)
true = np.array(true)

# THRESHOLD SEARCH
best_t = 0.5
best_acc = 0

val_probs, val_true = [], []

with torch.no_grad():
    for Xt, Xe, yb in val_loader:
        Xt, Xe = Xt.to(device), Xe.to(device)
        p = torch.sigmoid(model(Xt, Xe).squeeze())

        val_probs.extend(p.cpu().numpy())
        val_true.extend(yb.numpy())

val_probs = np.array(val_probs)
val_true = np.array(val_true)

for t in np.arange(0.3, 0.8, 0.02):
    preds_t = (val_probs > t).astype(int)

    acc = accuracy_score(val_true, preds_t)

    if acc > best_acc:
        best_acc = acc
        best_t = t

print("\nBest threshold:", best_t)

# FINAL METRICS
preds = (probs > best_t).astype(int)

print("\nFINAL RESULT")
print("Accuracy:", accuracy_score(true, preds))
print("Precision:", precision_score(true, preds))
print("Recall:", recall_score(true, preds))
print("F1:", f1_score(true, preds))
print("AUC:", roc_auc_score(true, probs))

cm = confusion_matrix(true, preds)
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

shuffled = np.copy(true)
np.random.shuffle(shuffled)

print("\nSANITY CHECK")
print("AUC (shuffled):", roc_auc_score(shuffled, probs))

fpr_dl, tpr_dl, _ = roc_curve(true, probs)
auc_dl = auc(fpr_dl, tpr_dl)

plt.figure()
plt.plot(fpr_dl, tpr_dl, label=f"DL (AUC={auc_dl:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Deep Learning")
plt.legend()
plt.grid(True)
plt.savefig("roc_dl.png")
plt.show()

plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig("loss_curve.png")
plt.show()

train_accs, val_accs = [], []
train_f1s, val_f1s = [], []

model.eval()

# TRAIN
train_preds, train_true_epoch = [], []

with torch.no_grad():
    for Xt, Xe, yb in train_loader:
        Xt, Xe = Xt.to(device), Xe.to(device)
        p = torch.sigmoid(model(Xt, Xe).squeeze())

        train_preds.extend((p.cpu().numpy() > best_t).astype(int))
        train_true_epoch.extend(yb.numpy())

train_acc = accuracy_score(train_true_epoch, train_preds)
train_f1 = f1_score(train_true_epoch, train_preds)

# VAL
val_preds, val_true_epoch = [], []

with torch.no_grad():
    for Xt, Xe, yb in val_loader:
        Xt, Xe = Xt.to(device), Xe.to(device)
        p = torch.sigmoid(model(Xt, Xe).squeeze())

        val_preds.extend((p.cpu().numpy() > best_t).astype(int))
        val_true_epoch.extend(yb.numpy())

val_acc = accuracy_score(val_true_epoch, val_preds)
val_f1 = f1_score(val_true_epoch, val_preds)


train_accs = [train_acc] * len(train_losses)
val_accs = [val_acc] * len(val_losses)

train_f1s = [train_f1] * len(train_losses)
val_f1s = [val_f1] * len(val_losses)

# ROC COMPARISON
plt.figure()

# DL
plt.plot(fpr_dl, tpr_dl, label=f"DL (AUC={auc_dl:.3f})")

# RF
try:
    rf_probs = np.load("rf_probs.npy")
    rf_true = np.load("rf_true.npy")

    if len(rf_probs) == len(true):
        fpr_rf, tpr_rf, _ = roc_curve(rf_true, rf_probs)
        plt.plot(fpr_rf, tpr_rf, label=f"RF (AUC={auc(fpr_rf, tpr_rf):.3f})")
    else:
        print("RF shape mismatch, skipped")

except:
    print("RF file not found")

# XGB
try:
    xgb_probs = np.load("xgb_probs.npy")
    xgb_true = np.load("xgb_true.npy")

    if len(xgb_probs) == len(true):
        fpr_xgb, tpr_xgb, _ = roc_curve(xgb_true, xgb_probs)
        plt.plot(fpr_xgb, tpr_xgb, label=f"XGB (AUC={auc(fpr_xgb, tpr_xgb):.3f})")
    else:
        print("XGB shape mismatch, skipped")

except:
    print("XGB file not found")

plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Comparison (DL vs RF vs XGB)")
plt.legend()
plt.grid(True)
plt.savefig("roc_comparison.png")
plt.show()

# F1 vs THRESHOLD CURVE
thresholds = np.arange(0.1, 0.9, 0.05)
f1_scores = []

for t in thresholds:
    preds_t = (probs > t).astype(int)
    f1_scores.append(f1_score(true, preds_t))

plt.figure()
plt.plot(thresholds, f1_scores, marker='o')
plt.xlabel("Threshold")
plt.ylabel("F1 Score")
plt.title("F1 Score vs Threshold")
plt.grid(True)
plt.savefig("f1_curve.png")
plt.show()