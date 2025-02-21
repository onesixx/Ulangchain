from rosie.log import logger
logger.info("fn_raptor.py!!!")

# RAPTOR의 특징:
# 재귀적 처리: RAPTOR는 텍스트를 임베딩, 클러스터링, 요약하는 과정을 재귀적으로 수행하여 트리 구조를 형성합니다.
# 계층적 구조: 이 트리 구조는 문서의 다양한 수준의 요약 정보를 포함하며, 넓은 주제적 이해와 세부적인 정보를 균형 있게 처리합니다.
# 추상화 수준: 추론 시 모델은 이 트리에서 정보를 검색하여 긴 문서의 다양한 추상화 수준의 정보를 통합할 수 있습니다
# 성능 향상: 특히 복잡한 다단계 추론이 필요한 질문 응답 작업에서 기존 방법보다 우수한 성능을 보입니다.
# 유연성: 추상적인 질문과 구체적인 질문 모두에 효과적으로 대응할 수 있습니다

# RAPTOR 장점
# 문서의 전체적인 맥락을 이해하고 다양한 수준의 정보를 통합하여 더 정확하고 포괄적인 답변을 제공할 수 있다
# 이는 기존의 검색 강화 언어 모델이 짧은 연속적인 텍스트 조각만을 검색하는 한계를 극복한 것입니다
import pickle
from typing import Optional, List, Tuple, Dict
import numpy as np
import pandas as pd
import umap.umap_ as umap
from sklearn.mixture import GaussianMixture
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from sklearn.cluster import MiniBatchKMeans

embd = OllamaEmbeddings(
    model='nomic-embed-text:latest',
    base_url="http://172.17.0.2:11434",
)

RANDOM_SEED = 42  # 재현성을 위한 고정된 시드 값

def global_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: Optional[int] = None,
    metric: str = "cosine",
) -> np.ndarray:
    """
    UMAP을 사용하여 임베딩의 전역 차원 축소를 수행합니다.
    args:
    - embeddings: numpy 배열로 된 입력 임베딩.
    - dim: 축소된 공간의 목표 차원 수.
    - n_neighbors: 선택사항; 각 점을 고려할 이웃의 수.(제공되지 않으면, default는 임베딩 수의 제곱)
    - metric: UMAP에 사용할 거리 측정 기준.
    returns:
    - 지정된 차원으로 축소된 임베딩의 numpy 배열.
    """
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    return umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=dim,
        metric=metric
    ).fit_transform(embeddings)

def local_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    num_neighbors: int = 10,
    metric: str = "cosine"
) -> np.ndarray:
    """
    임베딩에 대해 지역 차원 축소를 수행합니다. 이는 일반적으로 전역 클러스터링 이후에 사용됩니다.
    """
    return umap.UMAP(
        n_neighbors=num_neighbors,
        n_components=dim,
        metric=metric
    ).fit_transform(embeddings)

def get_optimal_clusters(
    embeddings: np.ndarray,
    max_clusters: int = 50,
    random_state: int = RANDOM_SEED
) -> int:
    """
    Gaussian Mixture Model을 사용하여 BIC(Bayesian 정보 기준)을 통해 최적의 클러스터 수를 결정합니다.
    args:
    - embeddings: numpy 배열로서의 입력 임베딩.
    - max_clusters: 고려할 최대 클러스터 수.
    - random_state: 재현성을 위한 시드.
    returns:
    - 발견된 최적의 클러스터 수를 나타내는 정수.
    """
    max_clusters = min( max_clusters, len(embeddings))  # 최대 클러스터 수와 임베딩의 길이 중 작은 값을 최대 클러스터 수로 설정
    n_clusters = np.arange(1, max_clusters)  # 1부터 최대 클러스터 수까지의 범위를 생성
    bics = []  # BIC 점수를 저장할 리스트
    for n in n_clusters:  # 각 클러스터 수에 대해 반복
        gm = GaussianMixture(
            n_components=n,
            random_state=random_state
        )  # 가우시안 혼합 모델 초기화
        gm.fit(embeddings)  # 임베딩에 대해 모델 학습
        bics.append(gm.bic(embeddings))  # 학습된 모델의 BIC 점수를 리스트에 추가
    return n_clusters[np.argmin(bics)]  # BIC 점수가 가장 낮은 클러스터 수를 반환

def GMM_cluster(embeddings: np.ndarray, threshold: float, random_state: int = 0):
    """
    확률 임계값을 기반으로 가우시안 혼합 모델(GMM)을 사용하여 임베딩을 클러스터링합니다.
    args:
    - embeddings: numpy 배열로서의 입력 임베딩.
    - threshold: 임베딩을 클러스터에 할당하기 위한 확률 임계값.
    - random_state: 재현성을 위한 시드.
    returns:
    - 클러스터 레이블과 결정된 클러스터 수를 포함하는 튜플.
    """

    # n_clusters = get_optimal_clusters(embeddings)  # 최적의 클러스터 수를 구합니다.
    # # 가우시안 혼합 모델을 초기화합니다.
    # gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    # gm.fit(embeddings)  # 임베딩에 대해 모델을 학습합니다.
    # probs = gm.predict_proba(
    #     embeddings
    # )  # 임베딩이 각 클러스터에 속할 확률을 예측합니다.
    # # 임계값을 초과하는 확률을 가진 클러스터를 레이블로 선택합니다.
    # labels = [np.where(prob > threshold)[0] for prob in probs]
    # MiniBatchKMeans 사용
    n_clusters = get_optimal_clusters(embeddings)
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(embeddings)

    return labels, n_clusters  # 레이블과 클러스터 수를 반환합니다.

def perform_clustering(
    embeddings: np.ndarray,
    dim: int,
    threshold: float,
) -> List[np.ndarray]:
    """
    임베딩에 대해 차원 축소, 가우시안 혼합 모델을 사용한 클러스터링, 각 글로벌 클러스터 내에서의 로컬 클러스터링을 순서대로 수행합니다.
    args:
    - embeddings: numpy 배열로 된 입력 임베딩입니다.
    - dim: UMAP 축소를 위한 목표 차원입니다.
    - threshold: GMM에서 임베딩을 클러스터에 할당하기 위한 확률 임계값입니다.
    returns:
    - 각 임베딩의 클러스터 ID를 포함하는 numpy 배열의 리스트입니다.
    """
    if len(embeddings) <= dim + 1:
        # 데이터가 충분하지 않을 때 클러스터링을 피합니다.
        return [np.array([0]) for _ in range(len(embeddings))]

    # 글로벌 차원 축소
    reduced_embeddings_global = global_cluster_embeddings(embeddings, dim)
    # 글로벌 클러스터링
    global_clusters, n_global_clusters = GMM_cluster(
        reduced_embeddings_global, threshold
    )

    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    # 각 글로벌 클러스터를 순회하며 로컬 클러스터링 수행
    for i in range(n_global_clusters):
        # 현재 글로벌 클러스터에 속하는 임베딩 추출
        global_cluster_embeddings_ = embeddings[
            np.array([i == gc for gc in global_clusters])  # in ==> ==
        ]

        if len(global_cluster_embeddings_) == 0:
            continue
        if len(global_cluster_embeddings_) <= dim + 1:
            # 작은 클러스터는 직접 할당으로 처리
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            # 로컬 차원 축소 및 클러스터링
            reduced_embeddings_local = local_cluster_embeddings(
                global_cluster_embeddings_, dim
            )
            local_clusters, n_local_clusters = GMM_cluster(
                reduced_embeddings_local, threshold
            )
        logger.info(f"Global cluster {i} has {n_local_clusters} local clusters {local_clusters}")
        # 로컬 클러스터 ID 할당, 이미 처리된 총 클러스터 수를 조정
        for j in range(n_local_clusters):
            local_cluster_embeddings_ = global_cluster_embeddings_[
                np.array([j == lc for lc in local_clusters])
            ]
            indices = np.where(
                (embeddings == local_cluster_embeddings_[:, None]).all(-1)
            )[1]
            for idx in indices:
                all_local_clusters[idx] = np.append(
                    all_local_clusters[idx], j + total_clusters
                )

        total_clusters += n_local_clusters

    return all_local_clusters

def embed(texts):
    # 텍스트 문서 목록에 대한 임베딩을 생성합니다.
    # 이 함수는 `embd` 객체가 존재한다고 가정하며, 이 객체는 텍스트 목록을 받아 그 임베딩을 반환하는 `embed_documents` 메소드를 가지고 있습니다.
    # args:
    # - texts: List[str], 임베딩할 텍스트 문서의 목록입니다.
    # returns:
    # - numpy.ndarray: 주어진 텍스트 문서들에 대한 임베딩 배열입니다.
    logger.info("Embedding texts...===> start")
    text_embeddings = embd.embed_documents(texts) # 텍스트 문서들의 임베딩을 생성합니다.

    #path = "/workspaces/wikidocs_251190/raptor_long2short/temp/text_embeddings.pkl"

    # with open(path, 'wb') as file:
    #     pickle.dump(text_embeddings, file)
    # with open(path, 'rb') as file:
    #     text_embeddings = pickle.load(file)

    logger.info("Embedding texts...<=== end")


    text_embeddings_np = np.array(text_embeddings)  # 임베딩을 numpy 배열로 변환합니다.
    return text_embeddings_np  # 임베딩된 numpy 배열을 반환합니다.




def embed_cluster_texts(texts):
    """
    텍스트 목록을 임베딩하고 클러스터링하여, 텍스트, 그들의 임베딩, 그리고 클러스터 라벨이 포함된 DataFrame을 반환합니다.
    이 함수는 임베딩 생성과 클러스터링을 단일 단계로 결합합니다. 임베딩에 대해 클러스터링을 수행하는 `perform_clustering` 함수의 사전 정의된 존재를 가정합니다.
    args:
    - texts: List[str], 처리될 텍스트 문서의 목록입니다.
    returns:
    - pandas.DataFrame: 원본 텍스트, 그들의 임베딩, 그리고 할당된 클러스터 라벨이 포함된 DataFrame입니다.
    """
    text_embeddings_np = embed(texts)  # 임베딩 생성
    cluster_labels = perform_clustering(
        text_embeddings_np, 10, 0.1, #force_all_finite=True
    )  # 임베딩에 대해 클러스터링 수행
    df = pd.DataFrame()  # 결과를 저장할 DataFrame 초기화
    df["text"] = texts  # 원본 텍스트 저장
    df["embd"] = list(text_embeddings_np)  # DataFrame에 리스트로 임베딩 저장
    df["cluster"] = cluster_labels  # 클러스터 라벨 저장
    return df

def fmt_txt(df: pd.DataFrame) -> str:
    """
    DataFrame에 있는 텍스트 문서를 단일 문자열로 포맷합니다.
    args:
    - df: 'text' 열에 포맷할 텍스트 문서가 포함된 DataFrame.
    returns:
    - 모든 텍스트 문서가 특정 구분자로 결합된 단일 문자열.
    """
    unique_txt = df["text"].tolist()  # 'text' 열의 모든 텍스트를 리스트로 변환
    return "--- --- \n --- --- ".join(
        unique_txt
    )  # 텍스트 문서들을 특정 구분자로 결합하여 반환

def embed_cluster_summarize_texts(
    texts: List[str],
    level: int,
    model: None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    텍스트 목록에 대해 임베딩, 클러스터링 및 요약을 수행합니다. 이 함수는 먼저 텍스트에 대한 임베딩을 생성하고,
    유사성을 기반으로 클러스터링을 수행한 다음, 클러스터 할당을 확장하여 처리를 용이하게 하고 각 클러스터 내의 내용을 요약합니다.
    args:
    - texts: 처리할 텍스트 문서 목록입니다.
    - level: 처리의 깊이나 세부 사항을 정의할 수 있는 정수 매개변수입니다.
    returns:
    - 두 개의 데이터프레임을 포함하는 튜플:
      1. 첫 번째 데이터프레임(`df_clusters`)은 원본 텍스트, 그들의 임베딩, 그리고 클러스터 할당을 포함합니다.
      2. 두 번째 데이터프레임(`df_summary`)은 각 클러스터에 대한 요약, 지정된 세부 수준, 그리고 클러스터 식별자를 포함합니다.
    """
    # 기본 모델 설정 (필요한 경우)
    if model is None:
        logger.info("====> Model is not provided, using default model")
        model = ChatOllama(
            base_url="http://172.17.0.2:11434",
            model="llama3.3:latest",
            temperature=0,
            streaming=True,
        )
    # 텍스트를 임베딩하고 클러스터링하여 'text', 'embd', 'cluster' 열이 있는 데이터프레임을 생성합니다.
    df_clusters = embed_cluster_texts(texts)

    # 클러스터를 쉽게 조작하기 위해 데이터프레임을 확장할 준비를 합니다.
    expanded_list = []

    # 데이터프레임 항목을 문서-클러스터 쌍으로 확장하여 처리를 간단하게 합니다.
    for index, row in df_clusters.iterrows():
        for cluster in row["cluster"]:
            expanded_list.append(
                {"text": row["text"], "embd": row["embd"], "cluster": cluster}
            )

    # 확장된 목록에서 새 데이터프레임을 생성합니다.
    expanded_df = pd.DataFrame(expanded_list)

    # 처리를 위해 고유한 클러스터 식별자를 검색합니다.
    all_clusters = expanded_df["cluster"].unique()

    # print(f"--Generated {len(all_clusters)} clusters--")

    # 요약
    # template = """여기 LangChain 표현 언어 문서의 하위 집합이 있습니다.
    # LangChain 표현 언어는 LangChain에서 체인을 구성하는 방법을 제공합니다.
    # 제공된 문서의 자세한 요약을 제공하십시오.
    template ="""
    여기 shiny for python 문서의 하위 집합이 있습니다.
    이 문서의 핵심 주제와 주요 포인트를 요약해 주세요:
    문서:
    {context}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model | StrOutputParser()

    # 각 클러스터 내의 텍스트를 요약을 위해 포맷팅합니다.
    summaries = []
    for i in all_clusters:
        df_cluster = expanded_df[expanded_df["cluster"] == i]
        formatted_txt = fmt_txt(df_cluster)
        summary = chain.invoke({"context": formatted_txt})
        summaries.append(summary)

    # 요약, 해당 클러스터 및 레벨을 저장할 데이터프레임을 생성합니다.
    df_summary = pd.DataFrame(
        {
            "summaries": summaries,
            "level": [level] * len(summaries),
            "cluster": list(all_clusters),
        }
    )

    return df_clusters, df_summary


def recursive_embed_cluster_summarize(
    texts: List[str], level: int = 1, n_levels: int = 3, model=None
) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    지정된 레벨까지 (또는 고유 클러스터의 수가 1이 될 때까지) 텍스트를
    재귀적으로 embedding, clustering, summarizing을 수행하여
    각 레벨에서의 결과를 저장합니다.
    args:
    - texts: List[str], 처리할 텍스트들. leaf_text
    - level: int, 현재 재귀 레벨 (1에서 시작).
    - n_levels: int, 재귀의 최대 깊이.
    returns:
    - Dict[int, Tuple[pd.DataFrame, pd.DataFrame]], 재귀 레벨을 키로 하고 해당 레벨에서의 클러스터 DataFrame과 요약 DataFrame을 포함하는 튜플을 값으로 하는 사전.
    """
    results = {}  # 각 레벨에서의 결과를 저장할 사전

    # 현재 레벨에 대해 임베딩, 클러스터링, 요약 수행
    df_clusters, df_summary = embed_cluster_summarize_texts(texts, level, model)

    # 현재 레벨의 결과 저장
    results[level] = (df_clusters, df_summary)

    # 추가 재귀가 가능하고 의미가 있는지 결정
    unique_clusters = df_summary["cluster"].nunique()
    if level < n_levels and unique_clusters > 1:
        # 다음 레벨의 재귀 입력 텍스트로 요약 사용
        new_texts = df_summary["summaries"].tolist()
        next_level_results = recursive_embed_cluster_summarize(
            new_texts, level + 1, n_levels
        )
        # 다음 레벨의 결과를 현재 결과 사전에 병합
        results.update(next_level_results)
    return results
