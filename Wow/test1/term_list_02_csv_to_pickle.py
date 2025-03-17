import pandas as pd
from pathlib import Path
import pickle

BASE_DIR = Path(__file__).resolve().parent

# 读取CSV文件
df = pd.read_csv(BASE_DIR.joinpath('SpellName.11.1.1.59324_lfonly.csv'))

# 提取Name_lang列并处理数据
names = (
    df['Name_lang']
    .dropna()  # 移除空值
    .loc[lambda x: ~x.str.contains(r'[()]')]  # 过滤包含括号的条目
    .loc[lambda x: ~x.str.contains(r'[\[\]]')]  # 过滤包含括号的条目
    .drop_duplicates()  # 去重
    .reset_index(drop=True)  # 重置索引
)

# # 保存到pickle文件
# names.to_pickle('filtered_names.pkl')
#
# # 打印结果
# print("去重后的名称列表：")
# print(names.to_frame(name='Filtered_Names'))
print("\n".join(names.tolist()))

pickle.dump(names.tolist(), open(BASE_DIR.joinpath("term_list.pkl"), "wb"))
