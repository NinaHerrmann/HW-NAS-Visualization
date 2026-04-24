import pandas as pd

# paths
all_path = "../all_cifar10_hwnas.csv"
res_path = "../results/result.csv"
out_path = "missing_cifar10.txt"

# load
all_df = pd.read_csv(all_path)
res_df = pd.read_csv(res_path)

# keep only cifar10 from all_hwnas (adjust value if your dataset name differs, e.g. "cifar-10")
all_c10 = all_df[all_df["dataset"].eq("cifar10")].copy()

# treat arch_index in all_hwnas as idx to compare to results
all_c10 = all_c10.rename(columns={"arch_index": "idx"})

# unique keys only (avoid duplicate lines)
all_keys = all_c10[["idx", "seed", "dataset"]].drop_duplicates()
res_keys = res_df[["idx", "seed", "dataset"]].drop_duplicates()

# find combinations in all_hwnas but not in result.csv
missing = all_keys.merge(res_keys, on=["idx", "seed", "dataset"], how="left", indicator=True)
missing = missing[missing["_merge"].eq("left_only")][["idx", "seed"]]

# write "idx seed" per line
missing.to_csv(out_path, sep=" ", header=False, index=False)

print(f"Wrote {len(missing)} combinations to {out_path}")