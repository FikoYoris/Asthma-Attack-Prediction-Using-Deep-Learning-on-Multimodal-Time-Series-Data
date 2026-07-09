import pandas as pd

df = pd.read_csv("data/processed/smartwatch_combined.csv")

df["datetime"] = pd.to_datetime(df["datetime"])

WINDOW = "15min"

rows = []

for user, df_u in df.groupby("user_key"):

    df_u = df_u.set_index("datetime")

    w = df_u.resample(WINDOW).agg({
        "hr": ["mean", "std", "min", "max"],
        "steps": ["sum", "mean"],
        "intensity": ["mean"]
    })

    w.columns = [
        "hr_mean",
        "hr_std",
        "hr_min",
        "hr_max",
        "steps_sum",
        "steps_mean",
        "intensity_mean"
    ]

    w = w.reset_index()
    w["user_key"] = user

    rows.append(w)

windowed = pd.concat(rows, ignore_index=True)

# handle missing
windowed["hr_std"] = windowed["hr_std"].fillna(0)
windowed = windowed.dropna(subset=["hr_mean"])

# save
windowed.to_csv("data/processed/smartwatch_15min.csv", index=False)

print("Windowed dataset:", windowed.shape)
