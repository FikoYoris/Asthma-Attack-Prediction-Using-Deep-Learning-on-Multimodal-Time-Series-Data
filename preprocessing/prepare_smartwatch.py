import pandas as pd

files = [
    "data/raw/anonym_aamos00_smartwatch1.csv",
    "data/raw/anonym_aamos00_smartwatch2.csv",
    "data/raw/anonym_aamos00_smartwatch3.csv",
]

dfs = []

for f in files:
    df = pd.read_csv(f)
    dfs.append(df)

sw = pd.concat(dfs, ignore_index=True)

# base date sintetis
base_date = pd.Timestamp("2020-01-01")

sw["datetime"] = base_date + pd.to_timedelta(sw["date"], unit="D")
sw["datetime"] = sw["datetime"] + pd.to_timedelta(sw["time"])

sw = sw.sort_values(["user_key", "datetime"])

print(sw.head())

sw.to_csv("data/processed/smartwatch_combined.csv", index=False)

print("Saved:", sw.shape)
