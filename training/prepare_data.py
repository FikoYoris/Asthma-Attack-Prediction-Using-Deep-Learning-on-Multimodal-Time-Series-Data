import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/processed/dataset_multimodal.csv")

print("Original shape:", df.shape)

df = df.sort_values(["user_key", "datetime"])
df = df.fillna(0)

print("\n TARGET DISTRIBUTION")
print(df["event_attack_window"].value_counts(normalize=True))

# SPLIT USER (ANTI-LEAKAGE)
users = df["user_key"].unique()

np.random.seed(42)
np.random.shuffle(users)

n = len(users)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

train_users = users[:train_end]
val_users = users[train_end:val_end]
test_users = users[val_end:]

train_df = df[df["user_key"].isin(train_users)].copy()
val_df = df[df["user_key"].isin(val_users)].copy()
test_df = df[df["user_key"].isin(test_users)].copy()

print("\nUSER SPLIT")
print("Train users:", len(train_users))
print("Val users:", len(val_users))
print("Test users:", len(test_users))

# VALIDATION CHECK NO DATA LEAKAGE
print("\nLEAKAGE CHECK")
print("Train ∩ Val:", len(set(train_users) & set(val_users)))
print("Train ∩ Test:", len(set(train_users) & set(test_users)))
print("Val ∩ Test:", len(set(val_users) & set(test_users)))

# SELECT FEATURES
time_cols = [
    "hr_mean","hr_std","hr_min","hr_max",
    "steps_sum","steps_mean","intensity_mean"
]

env_cols = [
    "temperature","humidity","pressure",
    "pm2_5","pm10","o3","no2","co"
]

target_col = "event_attack_window"

# PROCESS
def process_df(df):
    X_time = df[time_cols]
    X_env = df[env_cols]
    y = df[target_col]
    return X_time, X_env, y

X_train_time_df, X_train_env_df, y_train_df = process_df(train_df)
X_val_time_df, X_val_env_df, y_val_df = process_df(val_df)
X_test_time_df, X_test_env_df, y_test_df = process_df(test_df)

# SCALING (NO LEAKAGE)
scaler_time = StandardScaler()
scaler_env = StandardScaler()

X_train_time = scaler_time.fit_transform(X_train_time_df)
X_val_time = scaler_time.transform(X_val_time_df)
X_test_time = scaler_time.transform(X_test_time_df)

X_train_env = scaler_env.fit_transform(X_train_env_df)
X_val_env = scaler_env.transform(X_val_env_df)
X_test_env = scaler_env.transform(X_test_env_df)

# SEQUENCE
TIME_STEPS = 20
FUTURE_SHIFT = 1

def create_sequence(df, X_time, X_env, y):
    X_seq, X_env_seq, y_seq = [], [], []

    for user in df["user_key"].unique():
        idx = df["user_key"] == user

        Xt = X_time[idx]
        Xe = X_env[idx]
        yt = y[idx].reset_index(drop=True)

        for i in range(len(Xt) - TIME_STEPS - FUTURE_SHIFT):
            X_seq.append(Xt[i:i+TIME_STEPS])
            X_env_seq.append(Xe[i+TIME_STEPS])
            y_seq.append(yt.iloc[i+TIME_STEPS+FUTURE_SHIFT])

    return np.array(X_seq), np.array(X_env_seq), np.array(y_seq)

X_train_time, X_train_env, y_train = create_sequence(train_df, X_train_time, X_train_env, y_train_df)
X_val_time, X_val_env, y_val = create_sequence(val_df, X_val_time, X_val_env, y_val_df)
X_test_time, X_test_env, y_test = create_sequence(test_df, X_test_time, X_test_env, y_test_df)

print("\nFINAL SHAPE")
print("Train:", X_train_time.shape, X_train_env.shape)
print("Val:", X_val_time.shape, X_val_env.shape)
print("Test:", X_test_time.shape, X_test_env.shape)

print("\nLABEL DISTRIBUTION AFTER SEQUENCE")
print("Train:", np.mean(y_train))
print("Val:", np.mean(y_val))
print("Test:", np.mean(y_test))


np.save("data/processed/X_train_time.npy", X_train_time)
np.save("data/processed/X_train_env.npy", X_train_env)

np.save("data/processed/X_val_time.npy", X_val_time)
np.save("data/processed/X_val_env.npy", X_val_env)

np.save("data/processed/X_test_time.npy", X_test_time)
np.save("data/processed/X_test_env.npy", X_test_env)

np.save("data/processed/y_train.npy", y_train)
np.save("data/processed/y_val.npy", y_val)
np.save("data/processed/y_test.npy", y_test)

print("\nSaved MULTIMODAL dataset (train/val/test)")

import joblib

joblib.dump(scaler_time, "scaler_time.pkl")
joblib.dump(scaler_env, "scaler_env.pkl")