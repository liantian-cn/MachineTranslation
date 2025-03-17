import pandas as pd
from pathlib import Path
import pickle


def process_spell_files(en_file, cn_file):
    # 读取CSV文件并用空字符串填充NaN
    df_en = pd.read_csv(en_file).fillna('')
    df_cn = pd.read_csv(cn_file).fillna('')

    # 转换为字典列表
    en_list = df_en.to_dict('records')
    cn_list = df_cn.to_dict('records')

    merged_dict = {}

    # 处理英文数据
    for en_row in en_list:
        spell_id = en_row['ID']
        merged_dict[spell_id] = {
            'NameSubtext_en': en_row['NameSubtext_lang'],
            'Description_en': en_row['Description_lang'],
            'AuraDescription_en': en_row['AuraDescription_lang'],
            'NameSubtext_cn': '',
            'Description_cn': '',
            'AuraDescription_cn': ''
        }

    # 处理中文数据
    for cn_row in cn_list:
        spell_id = cn_row['ID']
        if spell_id in merged_dict:
            merged_dict[spell_id].update({
                'NameSubtext_cn': cn_row['NameSubtext_lang'],
                'Description_cn': cn_row['Description_lang'],
                'AuraDescription_cn': cn_row['AuraDescription_lang']
            })
        else:
            # 处理中文存在但英文不存在的情况
            merged_dict[spell_id] = {
                'NameSubtext_en': '',
                'Description_en': '',
                'AuraDescription_en': '',
                'NameSubtext_cn': cn_row['NameSubtext_lang'],
                'Description_cn': cn_row['Description_lang'],
                'AuraDescription_cn': cn_row['AuraDescription_lang']
            }

    return merged_dict


def generate_localization_list(merged_dict):
    localization_list = []

    for spell_id, data in merged_dict.items():
        # 处理NameSubtext
        if data['NameSubtext_en'] and data['NameSubtext_cn']:
            localization_list.append({
                'chinese': data['NameSubtext_cn'],
                'english': data['NameSubtext_en']
            })

        # 处理Description
        if data['Description_en'] and data['Description_cn']:
            localization_list.append({
                'chinese': data['Description_cn'],
                'english': data['Description_en']
            })

        # 处理AuraDescription
        if data['AuraDescription_en'] and data['AuraDescription_cn']:
            localization_list.append({
                'chinese': data['AuraDescription_cn'],
                'english': data['AuraDescription_en']
            })

    return localization_list


# 方法二：使用pandas去重（适合大数据量）
def deduplicate_list_pandas(result_list):
    df = pd.DataFrame(result_list)
    df.drop_duplicates(inplace=True)
    return df.to_dict('records')


# 使用示例
if __name__ == "__main__":
    # 第一段代码
    merged_data = process_spell_files(
        'Spell.11.1.0.59324_lfonly.csv',
        'Spell_zhCN.11.1.0.59324_lfonly.csv'
    )

    result_list = generate_localization_list(merged_data)

    result_list = deduplicate_list_pandas(result_list)

    BASE_DIR = Path(__file__).resolve().parent
    pickle.dump(result_list, open(BASE_DIR.joinpath("reference_list.pkl"), "wb"))

