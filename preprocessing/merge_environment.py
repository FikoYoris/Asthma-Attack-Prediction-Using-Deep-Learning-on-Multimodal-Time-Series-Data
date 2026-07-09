import pandas as pd
import numpy as np

sw = pd.read_csv("data/processed/smartwatch_15min.csv")
env = pd.read_csv("data/raw/anonym_aamos00_environment.csv")

print("SW columns BEFORE:", sw.columns)
print("ENV columns BEFORE:", env.columns)

# FIX DATE COLUMN
sw["datetime"] = pd.to_datetime(sw["datetime"])
sw["date"] = sw["datetime"].dt.dayofyear

# SORT DATA
sw = sw.sort_values(["user_key", "datetime"])
env = env.sort_values(["user_key", "date"])

print("\nSW columns AFTER:", sw.columns)
print("ENV columns AFTER:", env.columns)

# MERGE DATA
df = pd.merge(
    sw,
    env,
    on=["user_key", "date"],
    how="left"
)

print("\nAfter merge:", df.shape)

# FEATURE ENGINEERING
# TIME
df["hour"] = df["datetime"].dt.hour
df["is_night"] = ((df["hour"] >= 18) | (df["hour"] < 6)).astype(int)
df["is_day"] = ((df["hour"] >= 6) & (df["hour"] < 18)).astype(int)

# TREND
df["hr_diff"] = df.groupby("user_key")["hr_mean"].diff()
df["steps_diff"] = df.groupby("user_key")["steps_sum"].diff()

# ROLLING
df["hr_rolling_mean"] = (
    df.groupby("user_key")["hr_mean"]
    .rolling(3, min_periods=1)
    .mean()
    .reset_index(0, drop=True)
)

df["hr_rolling_std"] = (
    df.groupby("user_key")["hr_mean"]
    .rolling(3, min_periods=1)
    .std()
    .reset_index(0, drop=True)
)

# SPIKE 
df["hr_spike"] = (df["hr_max"] > df["hr_mean"] + 10).astype(int)

# ACTIVITY
df["activity_level"] = df["steps_sum"] / (df["hr_mean"] + 1)

# ANOMALY
df["hr_anomaly"] = (
    (df["hr_mean"] > df["hr_rolling_mean"] + df["hr_rolling_std"]) &
    (df["steps_sum"] < df["steps_sum"].rolling(3, min_periods=1).mean())
).astype(int)

df["night_anomaly"] = df["hr_anomaly"] * df["is_night"]
df["day_anomaly"] = df["hr_anomaly"] * df["is_day"]

# rolling anomaly
df["anomaly_count_1h"] = (
    df.groupby("user_key")["hr_anomaly"]
    .rolling(4, min_periods=1)
    .sum()
    .reset_index(0, drop=True)
)

df["anomaly_count_3h"] = (
    df.groupby("user_key")["hr_anomaly"]
    .rolling(12, min_periods=1)
    .sum()
    .reset_index(0, drop=True)
)

# STATISTICAL
df["hr_range"] = df["hr_max"] - df["hr_min"]
df["hr_cv"] = df["hr_std"] / (df["hr_mean"] + 1)
df["steps_intensity"] = df["steps_sum"] / (df["hr_std"] + 1)

# USER NORMALIZATION
df["hr_user_mean"] = df.groupby("user_key")["hr_mean"].transform("mean")
df["hr_user_std"] = df.groupby("user_key")["hr_mean"].transform("std")

df["hr_zscore"] = (
    (df["hr_mean"] - df["hr_user_mean"]) /
    (df["hr_user_std"] + 1e-5)
)

# LAG FEATURES
for lag in [1, 2, 4]:
    df[f"hr_mean_lag_{lag}"] = df.groupby("user_key")["hr_mean"].shift(lag)
    df[f"steps_lag_{lag}"] = df.groupby("user_key")["steps_sum"].shift(lag)

# INTERACTION FEATURES
df["pm25_hr"] = df["pm2_5"] * df["hr_zscore"]
df["humidity_hr"] = df["humidity"] * df["hr_zscore"]
df["temp_hr"] = df["temperature"] * df["hr_zscore"]

# EVENT LABEL
df["event_attack"] = (df["hr_zscore"] > 1.3).astype(int)

# EVENT WINDOW
df["event_attack_window"] = (
    df.groupby("user_key")["event_attack"]
    .rolling(12, min_periods=1)
    .max()
    .reset_index(0, drop=True)
)

# 6. HANDLE MISSING
df = df.fillna(0)

# 7. DEBUG
print("\nSample data:")
print(df.head())

print("\nColumns:")
print(df.columns)

print("\nFinal shape:", df.shape)

print("\nEVENT DISTRIBUTION:")
print(df["event_attack_window"].value_counts(normalize=True))

# SAVE
df.to_csv("data/processed/dataset_multimodal.csv", index=False)

print("\nSaved dataset_multimodal.csv")