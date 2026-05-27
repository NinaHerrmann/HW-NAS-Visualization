import pandas as pd

# paths
all_path = "../all_cifar10_hwnas.csv"
res_path = "../results/alldata.csv"
out_path = "missing_cifar10.txt"

# load
all_df = pd.read_csv(all_path)
res_df = pd.read_csv(res_path)

# keep only cifar10 from all_hwnas (adjust value if your dataset name differs, e.g. "cifar-10")
all_c10 = all_df[all_df["dataset"].eq("cifar10")].copy()
all_c10 = all_c10.rename(columns={"arch_index": "idx"})

# unique keys only (avoid duplicate lines)
all_keys = all_c10[["idx", "dataset"]].drop_duplicates()
res_keys = res_df[["idx", "dataset"]].drop_duplicates()

# find combinations in all_hwnas but not in result.csv
missing = all_keys.merge(res_keys, on=["idx", "dataset"], how="left", indicator=True)
missing = missing[missing["_merge"].eq("left_only")][["idx"]]

# write "idx seed" per line
missing.to_csv(out_path, sep=" ", header=False, index=False)
missing_idx = missing['idx']          # Series of ids to remove
df = pd.read_csv('../old_memory_results.csv')
df_clean = df[~df['idx'].isin(missing_idx)]
df_clean.to_csv('../memory_results.csv', index=False)
print(f"Wrote {len(missing)} combinations to {out_path}")
print(f"Wrote {(df_clean['idx'].nunique())} instead of {len(df)} diff = {len(df)-len(df_clean)}")