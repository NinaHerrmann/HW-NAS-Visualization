import pandas as pd

# paths
noseed = "noseed.txt"
noweight = "noweight.txt"
shouldwork = "shouldwork.txt"

df_noseed = pd.read_csv(f"rmdup/{noseed}.csv", header=None)
df_noweight = pd.read_csv(f"rmdup/{noweight}.csv", header=None)
df_shouldwork = pd.read_csv(f"rmdup/{shouldwork}.csv", header=None)
df_shonecolumn = df_shouldwork.iloc[:,0]
df_shonecolumn.drop_duplicates(inplace=True)

concatinated = pd.concat([df_noseed, df_shonecolumn])
print(len(df_shonecolumn))
print(len(df_noseed))

out = df_noseed.merge(df_shonecolumn, how='left', indicator=True)
print(out)

out = out[out['_merge'] == 'left_only'].drop(columns='_merge')
print(out)
