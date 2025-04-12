import pandas as pd

# 读取 totall_extend.csv 文件
df = pd.read_csv('./data/clearData/totall_extend.csv', header=None)

# 得到标签列索引
last_column_index = df.shape[1] - 1

# 统计原始标签数量
print("原始数据标签数量：")
print(df[last_column_index].value_counts())

# 对 BENIGN 类别最多保留 130000 条，随机抽取
benign_df = df[df[last_column_index] == 'BENIGN']
if len(benign_df) > 130000:
    benign_df = benign_df.sample(n=130000, random_state=42)

# 对其他类别随机抽取四分之一
other_df = df[df[last_column_index] != 'BENIGN']
sampled_other_df = other_df.groupby(last_column_index, group_keys=False).apply(lambda x: x.sample(frac=0.25, random_state=42))

# 合并筛选后的数据
final_df = pd.concat([benign_df, sampled_other_df])

# 保存为新的文件
final_df.to_csv('./data/clearData/totall_sampled.csv', index=False, header=False)

# 输出新的标签数量
print("\n新的数据标签数量：")
print(final_df[last_column_index].value_counts())