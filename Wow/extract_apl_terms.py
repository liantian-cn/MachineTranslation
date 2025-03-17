from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import re
from typing import List, Dict
import torch
import os

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

BASE_DIR = Path(__file__).parent
VECTOR_PATH = BASE_DIR / "reference_vectors.npy"
META_PATH = BASE_DIR / "reference_meta.json"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STOP_WORDS = {
    "if", "then", "else", "and", "or", "but",
    "the", "a", "an", "is", "in", "to", "for",
    "of", "by", "with", "set", "get", "name",
    "action", "use", "value", "var", "variable"
}


class AplTermExtractor:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        self.vectors = np.load(VECTOR_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def _preprocess_apl(self, text: str) -> str:
        """优化后的游戏文本预处理"""
        # 提取APL特有格式：保留缩写、带点号的属性路径
        tokens = re.findall(r"\b[a-zA-Z_]+(?:\.[a-z_]+)+\b|\b[a-zA-Z_]{3,}\b", text)

        # 精确过滤逻辑词，保留游戏术语
        filtered = [
            token.lower() for token in tokens
            if token.lower() not in STOP_WORDS
               and not re.match(r"^(?:name|use|set|value)$", token)
        ]

        return " ".join(sorted(set(filtered), key=filtered.index))  # 去重保序

    def find_references(self, input_text: str, top_n: int = 5) -> List[Dict]:
        """修正后的检索逻辑"""
        # 生成查询向量并归一化
        query = self._preprocess_apl(input_text)
        query_vec = self.model.encode(query, convert_to_tensor=True, device=DEVICE)
        query_vec = query_vec / query_vec.norm()  # 添加归一化

        # 归一化参考向量（重要！）
        reference_norm = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        normalized_ref = self.vectors / reference_norm

        # 计算余弦相似度
        scores = np.dot(normalized_ref, query_vec.cpu().numpy())

        # 关联分数和元数据
        top_indices = np.argsort(scores)[-top_n:][::-1]
        return [{**self.metadata[idx], "score": float(scores[idx])} for idx in top_indices]


# 分步骤执行示例
def main():
    # Step 1: 初始化引擎
    print("Loading engine...")
    extractor = AplTermExtractor()

    # Step 2: 处理输入文本
    sample_apl = """
    # chaos_strike
    """

    print("\nExtracting references:")
    results = extractor.find_references(sample_apl, top_n=20)

    # 输出结果
    for idx, item in enumerate(results, 1):
        print(f"\n#{idx} [Score: {item.get('score', 0):.3f}]")
        print(f"Name: {item['Name_enUS']} → {item['Name_zhCN']}")
        print(f"Full: {item.get('FullString_zhCN', '')}")
        print(item)


if __name__ == "__main__":
    main()

"""

我想用大语言模型帮我翻译一些游戏文本。

但首先，是这些文本里有大量的游戏内专有词汇，翻译为中文很困难。但我现在有大量的中英文对照参考，存在于一个List[Dict]的结构中，
{
"Name_enUS": "",
"Name_zhCN": "",
"FullString_enUS":"",
"FullString_zhCN":"",
}

总数有数十万条，一次投入大语言模型肯定是不行的，如何将这些参考向量化。我要实现的，是根据翻译样本，提取相关的内容作为参考。
特别说明，从参考意义上来说，Name_enUS的权重要远高于FullString_enUS，如果让我量化，那就是20倍。

我想分2个文件完成这件事，
vectorize_references.py：将这些的中英文对照参考向量话，然后存为文件，以备将来使用。
extract_apl_terms.py：根据APL样例，提取参照词汇的，提取数量可以控制。

可喜可贺的是，vectorize_references我已经写完了，最终生成了reference_vectors.npy和reference_meta.json。这个文件的全部代码在下方的[pre_code]标签里。
请仔细阅读这部分代码，理解我之前做了什么。

现在需要帮我写下一个文件extract_apl_terms.py，这个文件根据游戏文本样例，提取参照词汇的。

- 游戏文本样例：

# Aldrachi Reaver
actions.ar=variable,name=rg_inc,op=set,value=buff.rending_strike.down&buff.glaive_flurry.up&cooldown.blade_dance.up&gcd.remains=0|variable.rg_inc&prev_gcd.1.death_sweep
actions.ar+=/pick_up_fragment,use_off_gcd=1,if=fury<=90
actions.ar+=/variable,name=fel_barrage,op=set,value=talent.fel_barrage&(cooldown.fel_barrage.remains<gcd.max*7&(active_enemies>=desired_targets+raid_event.adds.count|raid_event.adds.in<gcd.max*7|raid_event.adds.in>90)&(cooldown.metamorphosis.remains|active_enemies>2)|buff.fel_barrage.up)&!(active_enemies=1&!raid_event.adds.exists)
actions.ar+=/chaos_strike,if=buff.rending_strike.up&buff.glaive_flurry.up&(variable.rg_ds=2|active_enemies>2)&time>10
actions.ar+=/annihilation,if=buff.rending_strike.up&buff.glaive_flurry.up&(variable.rg_ds=2|active_enemies>2)
actions.ar+=/reavers_glaive,if=buff.glaive_flurry.down&buff.rending_strike.down&buff.thrill_of_the_fight_damage.remains<gcd.max*4+(variable.rg_ds=2)+(cooldown.the_hunt.remains<gcd.max*3)*3+(cooldown.eye_beam.remains<gcd.max*3&talent.shattered_destiny)*3&(variable.rg_ds=0|variable.rg_ds=1&cooldown.blade_dance.up|variable.rg_ds=2&cooldown.blade_dance.remains)&(buff.thrill_of_the_fight_damage.up|!prev_gcd.1.death_sweep|!variable.rg_inc)&active_enemies<3&!action.reavers_glaive.last_used<5&debuff.essence_break.down&(buff.metamorphosis.remains>2|cooldown.eye_beam.remains<10|fight_remains<10)
actions.ar+=/reavers_glaive,if=buff.glaive_flurry.down&buff.rending_strike.down&buff.thrill_of_the_fight_damage.remains<4&(buff.thrill_of_the_fight_damage.up|!prev_gcd.1.death_sweep|!variable.rg_inc)&active_enemies>2|fight_remains<10
actions.ar+=/call_action_list,name=ar_cooldown
actions.ar+=/run_action_list,name=ar_opener,if=(cooldown.eye_beam.up|cooldown.metamorphosis.up|cooldown.essence_break.up)&time<15&(raid_event.adds.in>40)
actions.ar+=/sigil_of_spite,if=debuff.essence_break.down&debuff.reavers_mark.remains>=2-talent.quickened_sigils
actions.ar+=/run_action_list,name=ar_fel_barrage,if=variable.fel_barrage&raid_event.adds.up

- 这些游戏文本不是常用的语法，请分析这些游戏文本的语法特点，然后决定如何处理。
 -  比如：要过滤一些词汇，比如   "if", "then", "else", "and", "or", "but",            "the", "a", "an", "is", "in", "to", "for",            "of", "by", "with", "set", "get", "name",            "action", "use", "value", "var", "variable"
- 提取数量可以控制。
- 文件应该分为2个步骤
  1. 加载引擎，读取向量和原文件。
  2. 输入文本，输出参考文本。
  3. 步骤2可分次执行。



[pre_code]

class Vectorizer:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        self.model.max_seq_length = 256  # 缩短序列长度提升效率

    def process_batch(self, batch):

        names = [x.get("Name_enUS", "") for x in batch]
        fulls = [x.get("FullString_enUS", "") for x in batch]

        # 合并文本并编码
        embeddings = self.model.encode(
            names + fulls,
            convert_to_tensor=True,
            show_progress_bar=False,
            device=DEVICE
        )

        # 分割Name和FullString的向量
        name_embs = embeddings[:len(batch)]
        full_embs = embeddings[len(batch):]

        # 计算加权平均
        weighted_vecs = (name_embs * NAME_WEIGHT + full_embs) / (NAME_WEIGHT + 1)
        return weighted_vecs.cpu().numpy()


def main():
    print(f"Using device: {DEVICE}")

    # 读取原始数据
    with open(BASE_DIR / "translations.json", "r", encoding="utf-8") as f:
        all_data = json.load(f)

    # 初始化处理器
    processor = Vectorizer()

    # 预分配内存空间
    all_vectors = np.zeros((len(all_data), 768), dtype=np.float32)  # mpnet-base维度为768
    meta_data = []

    # 分批次处理
    for batch_idx in tqdm(
            range(0, len(all_data), BATCH_SIZE),
            desc="Vectorizing",
            unit="batch"
    ):
        batch = all_data[batch_idx: batch_idx + BATCH_SIZE]
        vecs = processor.process_batch(batch)

        # 存储结果
        start_idx = batch_idx
        end_idx = start_idx + len(batch)
        all_vectors[start_idx:end_idx] = vecs
        meta_data.extend(batch)

    # 保存文件
    np.save(BASE_DIR / "reference_vectors.npy", all_vectors)
    with open(BASE_DIR / "reference_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(all_data)} vectors and metadata")


if __name__ == "__main__":
    main()

[/pre_code]


"""
