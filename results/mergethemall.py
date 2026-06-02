import pandas as pd

df_memory_modelsize = pd.read_csv("../intermediateresults/model_sizes.csv")
df_memory_modelsize.drop_duplicates(inplace=True)

df_hwnas = pd.read_csv("../intermediateresults/all_hwnas.csv", index_col=0)
df_hwnas = df_hwnas[df_hwnas["dataset"] == "cifar10"]
df_hwnas.drop_duplicates(inplace=True)

df_accuracy = pd.read_csv("../intermediateresults/result.csv")
df_accuracy.drop_duplicates(inplace=True)
df_accuracy = df_accuracy[df_accuracy["dataset"] == "cifar10"]

tmp = pd.merge(df_accuracy,
               df_memory_modelsize,
               on=['idx', 'seed'],
               how='inner')

hwnas_keep = df_hwnas[['seed','arch_index',
                       'arch_0','arch_1','arch_2',
                       'arch_3','arch_4','arch_5',
                       'dataset']]
merged = pd.merge(tmp,
                  hwnas_keep, left_on=['idx', 'seed', 'dataset'], right_on=['arch_index', 'seed', 'dataset'],
                  how='inner')

cols = ['arch_0','arch_1','arch_2','arch_3','arch_4','arch_5',
        'dataset','espdl_size_bytes','torch_size_bytes',
        'accuracy_fp32','accuracy_esp', 'idx', 'seed']
final_df = merged[cols]

print(final_df['idx'].nunique())
final_df.to_csv("./Lennart.csv")