import json

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


if __name__ == "__main__":
    spell_id =[22812, 5487, 106951, 768, 33786, 1850, 339, 209749, 205636, 274283, 202770, 6795, 274282, 319454, 2637, 99, 29166, 192081, 22570, 33917, 1126, 102359, 5211, 8921, 124974, 274281, 5215, 20484, 8936, 774, 2782, 108238, 106839, 78675, 2908, 210053, 106898, 191034, 78674, 202347, 93402, 18562, 305497, 252216, 783, 114282, 132469, 102793, 202425, 48438, 88747, 190984]
    spell_id = list(set(spell_id))


    with open(BASE_DIR.joinpath('translations.json'), 'r', encoding='utf-8') as f:
        data = json.load(f)

    spell_set = set(spell_id)
    result = []
    for item in data:
        try:
            # 尝试将 ID 转换为整数
            item_id = int(item['ID'])
            if item_id in spell_set:
                # 如果 ID 存在于 spell_list 中，添加到结果中
                result.append(item)
        except (KeyError, ValueError):
            # 如果有异常（如缺少 ID 或者类型转换失败），则跳过该条目
            continue

    # 打印结果
    # print(result)

    for res in result:
        # print(f"{res['FullString_enUS']} => {res['FullString_zhCN']}")
        print(f"elseif ability_id == {res['ID']}  then\n    return Cast(\"{res['Name_zhCN']}\")")