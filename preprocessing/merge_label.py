import pandas as pd

df = pd.read_csv("data/processed/dataset_multimodal.csv")

labels = pd.read_csv("data/processed/labels.csv")

# merge
final_df = pd.merge(
    df,
    labels,
    on=["user_key", "date"],
    how="left"
)

print(final_df.head())
print("Shape sebelum drop:", final_df.shape)

# cek missing label
print("\nMissing label BEFORE drop:")
print(final_df["asthma_attack"].isna().sum())

# drop data tanpa sample
final_df = final_df.dropna(subset=["asthma_attack"])

# cek ulang setelah drop
print("\nShape setelah drop:", final_df.shape)

print("\nMissing label AFTER drop:")
print(final_df["asthma_attack"].isna().sum())

# cek distribusi label
print("\nLabel distribution:")
print(final_df["asthma_attack"].value_counts())

# simpan dataset final
final_df.to_csv("data/processed/final_dataset.csv", index=False)

print("\nSaved final_dataset.csv")
