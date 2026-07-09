import pandas as pd

files = [
    "data/raw/anonym_aamos00_smartwatch1.csv",
    "data/raw/anonym_aamos00_smartwatch2.csv",
    "data/raw/anonym_aamos00_smartwatch3.csv",
]

dfs = []

for f in files:
    df = pd.read_csv(f)
    print("FILE:", f)

    print("\nHEAD:")
    print(df.head())

    print("\nCOLUMNS:")
    print(df.columns)

    dfs.append(df)

# menggabungkan semua file
sw = pd.concat(dfs, ignore_index=True)

print("\nTOTAL ROWS:", len(sw))

print("\nMISSING VALUES:")
print(sw.isna().sum())
