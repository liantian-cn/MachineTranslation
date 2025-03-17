import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import faiss
import spacy


# 初始化模型和工具（步骤2、3）
class TranslationReferenceSystem:
    def __init__(self, reference_list, term_list):
        self.reference_list = reference_list
        self.term_list = term_list

        # 加载语义模型
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.nlp = spacy.load("en_core_web_sm")

        # 预处理参照数据（步骤2）
        self._prepare_reference_index()

        # 预处理专有名词（步骤3）
        self._prepare_term_embeddings()

    def _prepare_reference_index(self):
        """建立参照语句的向量索引"""
        # 生成所有英文句子的向量
        english_sentences = [item["english"] for item in self.reference_list]
        self.reference_embeddings = self.embedding_model.encode(english_sentences,
                                                                convert_to_tensor=False)

        # 创建FAISS索引
        self.index = faiss.IndexFlatL2(self.reference_embeddings.shape[1])
        self.index.add(self.reference_embeddings.astype(np.float32))

    def _prepare_term_embeddings(self):
        """生成专有名词的向量"""
        self.term_embeddings = self.embedding_model.encode(self.term_list,
                                                           convert_to_tensor=False)

    def _extract_high_weight_words(self, text, threshold=0.7):
        """提取高权重词（步骤4）"""
        # 分词处理
        doc = self.nlp(text)
        tokens = [token.text.lower() for token in doc if not token.is_stop and token.is_alpha]

        # 生成文本词的向量
        token_embeddings = self.embedding_model.encode(tokens, convert_to_tensor=False)

        # 计算与专有名词的相似度
        similarities = cosine_similarity(token_embeddings, self.term_embeddings)

        # 找出相似度超过阈值的词
        high_weight_words = set()
        for i, token in enumerate(tokens):
            max_sim = np.max(similarities[i])
            if max_sim > threshold:
                # 获取最相似的专有名词
                closest_term_idx = np.argmax(similarities[i])
                closest_term = self.term_list[closest_term_idx]
                high_weight_words.add(closest_term)
                high_weight_words.add(token)

        return list(high_weight_words)

    def get_reference_sentences(self, text, top_n=5):
        """获取参照语句（主入口）"""
        # 步骤4：提取高权重词
        high_weight_words = self._extract_high_weight_words(text)

        # 步骤5：收集参照语句
        references = []
        for word in high_weight_words:
            # 找到包含该词的参照语句
            word_embedding = self.embedding_model.encode([word], convert_to_tensor=False)

            # 使用FAISS搜索相似句子
            distances, indices = self.index.search(word_embedding.astype(np.float32), top_n)

            # 收集结果
            for idx in indices[0]:
                references.append(self.reference_list[idx])

        # 去重并返回
        unique_refs = {(r['chinese'], r['english']): r for r in references}.values()
        return list(unique_refs)


# 使用示例
if __name__ == "__main__":
    # 示例数据

    import pickle
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent
    # reference_list = [
    #     {"chinese": "使用暗影步接近敌人", "english": "Use Shadowstep to approach enemies"},
    #     {"chinese": "发动火焰冲击", "english": "Cast Fireblast"},
    #     # ... 其他30万条数据
    # ]
    reference_list = pickle.load(open(BASE_DIR.joinpath("reference_list.pkl"), "rb"))
    # term_list = ["Shadowstep", "Fireblast", "Pyroblast"]
    term_list = pickle.load(open(BASE_DIR.joinpath("term_list.pkl"), "rb"))

    # 初始化系统
    system = TranslationReferenceSystem(reference_list, term_list)

    # 待翻译文本
    input_text = "When facing melee enemies, first use Shadowstep to create distance, then cast Fireblast."

    # 获取参照语句
    references = system.get_reference_sentences(input_text)
    print("找到的参照语句：")
    for ref in references:
        print(f"- {ref['english']} => {ref['chinese']}")
