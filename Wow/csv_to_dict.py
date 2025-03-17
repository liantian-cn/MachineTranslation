import json
import pandas as pd
from pathlib import Path
from typing import List, Dict

BASE_DIR = Path(__file__).resolve().parent


def process_spell_data(base_dir: Path) -> List[Dict]:
    """
    处理技能数据并生成结构化字典列表

    Args:
        base_dir: 包含四个CSV文件的目录路径

    Returns:
        结构化的字典列表，只包含有效的技能数据
    """
    # 定义文件路径
    spell_en = base_dir / "Spell.csv"
    spell_zh = base_dir / "Spell_zhCN.csv"
    name_en = base_dir / "SpellName.csv"
    name_zh = base_dir / "SpellName_zhCN.csv"

    # 加载Spell英文数据
    spell_df_en = pd.read_csv(
        spell_en,
        usecols=["ID", "NameSubtext_lang", "Description_lang", "AuraDescription_lang"],
        dtype=str
    ).add_suffix("_enUS").rename(columns={"ID_enUS": "ID"})

    # 加载Spell中文数据
    spell_df_zh = pd.read_csv(
        spell_zh,
        usecols=["ID", "NameSubtext_lang", "Description_lang", "AuraDescription_lang"],
        dtype=str
    ).add_suffix("_zhCN").rename(columns={"ID_zhCN": "ID"})

    # 合并Spell中英文数据
    spell_df = pd.merge(spell_df_en, spell_df_zh, on="ID", how="inner")

    # 加载技能名称英文数据
    name_df_en = pd.read_csv(
        name_en,
        usecols=["ID", "Name_lang"],
        dtype=str
    ).add_suffix("_enUS").rename(columns={"ID_enUS": "ID", "Name_lang_enUS": "Name_enUS"})

    # 加载技能名称中文数据
    name_df_zh = pd.read_csv(
        name_zh,
        usecols=["ID", "Name_lang"],
        dtype=str
    ).add_suffix("_zhCN").rename(columns={"ID_zhCN": "ID", "Name_lang_zhCN": "Name_zhCN"})

    # 合并技能名称中英文数据
    name_df = pd.merge(name_df_en, name_df_zh, on="ID", how="inner")

    # 最终合并所有数据
    merged_df = pd.merge(spell_df, name_df, on="ID", how="inner")

    # 过滤包含OLD/TEST的名称
    filter_cond = (
            ~merged_df["Name_enUS"].fillna("").str.contains("OLD|TEST|DNT") &
            ~merged_df["Name_zhCN"].fillna("").str.contains("OLD|TEST|DNT")
    )
    merged_df = merged_df[filter_cond]

    # 填充NaN为空白字符串
    merged_df = merged_df.fillna("")

    # 生成完整字符串字段
    merged_df["FullString_enUS"] = (
            merged_df["Name_enUS"] + ":" +
            merged_df["NameSubtext_lang_enUS"] + "," +
            merged_df["Description_lang_enUS"] + "," +
            merged_df["AuraDescription_lang_enUS"]
    )
    merged_df["FullString_zhCN"] = (
            merged_df["Name_zhCN"] + ":" +
            merged_df["NameSubtext_lang_zhCN"] + "," +
            merged_df["Description_lang_zhCN"] + "," +
            merged_df["AuraDescription_lang_zhCN"]
    )

    # 转换字段名称
    renamed_df = merged_df.rename(columns={
        "NameSubtext_lang_enUS": "NameSubtext_enUS",
        "NameSubtext_lang_zhCN": "NameSubtext_zhCN",
        "Description_lang_enUS": "Description_enUS",
        "Description_lang_zhCN": "Description_zhCN",
        "AuraDescription_lang_enUS": "AuraDescription_enUS",
        "AuraDescription_lang_zhCN": "AuraDescription_zhCN"
    })

    # 生成最终字典列表
    return [
        {"ID": row["ID"], **{k: v for k, v in row.items() if k != "ID"}}
        for row in renamed_df.to_dict(orient="records")
    ]


def deduplicate_by_key(data: List[Dict], keys: list) -> List[Dict]:
    """根据指定字段组合去重字典列表

    Args:
        data: 需要去重的字典列表
        keys: 用于判断重复的字段组合，如 ['id', 'name']

    Returns:
        去重后的新列表，保留首次出现顺序
    """
    seen = set()
    result = []
    for d in data:
        fingerprint = tuple(d.get(k) for k in keys)
        if fingerprint not in seen:
            seen.add(fingerprint)
            result.append(d)
    return result


def filter_dicts(data: list[dict]) -> list[dict]:
    """过滤FullString中文和英文内容相同的字典。

    Args:
        data: 包含字典的列表，每个字典至少包含FullString_zhCN和FullString_enUS键

    Returns:
        过滤后的新列表，仅保留两种语言版本文案不同的条目
    """
    return [d for d in data if d["FullString_zhCN"] != d["FullString_enUS"]]


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    spell_data = process_spell_data(BASE_DIR)
    spell_data = deduplicate_by_key(spell_data, ["FullString_enUS"])
    spell_data = filter_dicts(spell_data)



    print(spell_data)
    json.dump(spell_data, open("translations.json", "w", encoding="utf-8"), ensure_ascii=False, indent=4)


"""

我有4个csv文件，其中为中英文对照的两组。
Spell.csv
Spell_zhCN.csv
SpellName.csv
SpellName_zhCN.csv

Spell的内容格式为
ID,NameSubtext_lang,Description_lang,AuraDescription_lang

SpellName的内容格式为
ID,Name_lang

可以理解为SpellName是技能名称，Spell为技能的其他一些设定，ID相同的是一组技能。

帮我写一个Python脚本。

生成一个由dict组成的list。dict的结构为

{
"ID":"{技能ID}":
"Name_enUS":"SpellName中的Name_lang",
"Name_zhCN":"SpellName_zhCN中的Name_lang",
"NameSubtext_enUS":"Spell中的NameSubtext_lan",
"NameSubtext_zhCN":"Spell_zhCN中的NameSubtext_lang",
"Description_enUS":"Spell中的Description_lang",
"Description_zhCN":"Spell_zhCN中的Descriptiont_lang",
"AuraDescription_enUS":"Spell中的AuraDescription_lang",
"AuraDescription_zhCN":"Spell_zhCN中的AuraDescription_lang",
"FullString_enUS":"{Name_enUS}:{NameSubtext_enUS},{Description_enUS},{AuraDescription_enUS}",
"FullString_zhCN":"{Name_zhCN}:{NameSubtext_zhCN},{Description_zhCN},{AuraDescription_zhCN}",
}

存在以下问题
1. ID不是连续的.
2. 某个ID可能在其中某个文件有，其他文件没有，仅当4个文件都有的时候，才有效。
3. NameSubtext_lang,Description_lang,AuraDescription_lang可能为空，空则保存成""
4. 过滤掉Name中包含OLD和TEST。OLD和TEST肯定是大写。
5. 每个文件都十分巨大（100万行），我强烈建议使用pandas.read_csv()读取，然后修改表头，再以ID为主键合并四个df。
6. 所有文件都在BASE_DIR下，BASE_DIR是pathlib.Path对象。



"""