import json
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

# 配置参数
BASE_DIR = Path(__file__).resolve().parent  # 确保已预先配置好路径
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 512  # 根据GPU显存调整
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
NAME_WEIGHT = 20  # Name字段权重


class Vectorizer:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        self.model.max_seq_length = 256  # 缩短序列长度提升效率

    def process_batch(self, batch):
        """处理单个批次数据"""
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
extract_apl_terms.py：根据APL样例，提取参照词汇的，提取数量可以控制，暂时不写出这个函数。

vectorize_references.py的要求如下
- 所有文件都在BASE_DIR下，BASE_DIR是pathlib.Path对象，已经预先设置。
- 因为存在大量词典中没有的专有名词，比如Aldrachi。如果使用任何向量化模型或者其他方法。针对这种没有见过的词汇，反而需要增加权重。绝对不能忽略。
- 如果可以使用cuda就用，不强求。
- 由于耗时很长，尽量给出进度。

参考步骤
0. 读取translations.json文件，得到List[Dict]对象。
1. 加载模型（可选）
2. 批处理读取所有enUS文本，生成嵌入向量。
3. 将向量和对应的中英文保存到文件。可能保存为两部分：一个文件存储所有向量（格式根据方法而定）。另一个文件存储对应的原始字典数据（json格式）。





"""
