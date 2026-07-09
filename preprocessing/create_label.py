import pandas as pd

dq = pd.read_csv("data/raw/anonym_aamos00_dailyquestionnaire.csv")

# convert boolean
dq["daily_night_symp"] = dq["daily_night_symp"].astype(bool)
dq["daily_day_symp"] = dq["daily_day_symp"].astype(bool)
dq["daily_limit_activity"] = dq["daily_limit_activity"].astype(bool)

# buat label
dq["asthma_attack"] = (
    (dq["daily_night_symp"] == True) |
    (dq["daily_day_symp"] == True) |
    (dq["daily_limit_activity"] == True) |
    (dq["daily_relief_inhaler"] >= 2)
).astype(int)

print(dq[["user_key", "date", "asthma_attack"]].head())

# simpan label
dq[["user_key", "date", "asthma_attack"]].to_csv(
    "data/processed/labels.csv",
    index=False
)

print("Saved labels.csv")
