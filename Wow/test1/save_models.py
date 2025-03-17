import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import spacy
import pickle
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


class ModelPersister:
    def __init__(self, reference_list, term_list):
        self.reference_list = reference_list
        self.term_list = term_list

        # 初始化模型
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.nlp = spacy.load("en_core_web_sm")

        # 预处理数据
        self._prepare_data()

    def _prepare_data(self):
        """预处理并保存所有必要数据"""
        # 生成并保存参考语句索引
        english_sentences = [item["english"] for item in self.reference_list]
        reference_embeddings = self.embedding_model.encode(english_sentences, convert_to_tensor=False)

        index = faiss.IndexFlatL2(reference_embeddings.shape[1])
        index.add(reference_embeddings.astype(np.float32))
        faiss.write_index(index, "faiss_index.bin")

        # 生成并保存术语嵌入
        term_embeddings = self.embedding_model.encode(self.term_list, convert_to_tensor=False)
        np.save(BASE_DIR.joinpath("term_embeddings.npy"), term_embeddings)

        # 保存模型
        self.embedding_model.save("sbert_model")


if __name__ == "__main__":
    reference_list = pickle.load(open(BASE_DIR.joinpath("reference_list.pkl"), "rb"))

    term_list = pickle.load(open(BASE_DIR.joinpath("term_list.pkl"), "rb"))

    persister = ModelPersister(reference_list, term_list)
