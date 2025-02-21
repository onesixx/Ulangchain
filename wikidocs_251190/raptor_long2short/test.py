import numpy as np
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModel
import torch

# 필요한 라이브러리 설치
# pip install transformers torch numpy scikit-learn

class RAPTOR:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def get_embeddings(self, texts):
        # 문장 임베딩을 생성
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()  # [CLS] 토큰의 임베딩 사용
        return embeddings

    def cluster_and_summarize(self, texts):
        embeddings = self.get_embeddings(texts)

        # 클러스터링 - KMeans를 사용
        n_clusters = min(5, len(texts))  # 클러스터 수는 텍스트 수에 따라 조정
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
        clusters = kmeans.labels_

        # 클러스터별 요약 (여기서는 간단히 첫 번째 문장 사용)
        summaries = []
        for cluster_id in range(n_clusters):
            cluster_texts = [texts[i] for i in range(len(texts)) if clusters[i] == cluster_id]
            if cluster_texts:
                summaries.append(cluster_texts[0])  # 단순히 첫 번째 문장을 요약으로 사용
            else:
                summaries.append("No text in this cluster.")

        return summaries, clusters

    def build_tree(self, texts, depth=2):
        if depth == 0 or len(texts) <= 1:
            return texts  # 리프 노드

        summaries, clusters = self.cluster_and_summarize(texts)
        tree = []

        for i, summary in enumerate(summaries):
            subtexts = [texts[j] for j in range(len(texts)) if clusters[j] == i]
            tree.append({
                'summary': summary,
                'children': self.build_tree(subtexts, depth - 1)
            })

        return tree

# 예제 사용
texts = [
    "This is the first text.",
    "Another text to process.",
    "A third text for clustering.",
    "Fourth one here.",
    "Yet another text to add.",
]

raptor = RAPTOR()
tree = raptor.build_tree(texts)

# 트리 구조 출력
import json
print(json.dumps(tree, indent=2))