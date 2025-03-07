import numpy as np
np.random.seed(1234)             # make reproducible

# ------ Getting some data ------
d = 64                           # dimension
nb = 100000                      # database size
nq = 10000                       # nb of queries

xb = np.random.random((nb, d)).astype('float32')  # database
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')  # query
xq[:, 0] += np.arange(nq) / 1000.

# ------ Building an index and adding the vectors to it ------
import faiss                   # make faiss available
res = faiss.StandardGpuResources()  # use a single GPU
index = faiss.IndexFlatL2(d)   # build the index
gpu_index = faiss.index_cpu_to_gpu(res, 0, index) # build the index
print(index.is_trained)
index.add(xb)                  # add vectors to the index
print(index.ntotal)

# ------ Searching (Querying the index) ------
k = 4                          # we want to see 4 nearest neighbors
D, I = index.search(xb[:5], k) # sanity check
print(I)     # an array of indices
print(D)     # distances

D, I = index.search(xq, k)     # actual search
print(I[:5])                   # neighbors of the 5 first queries
print(I[-5:])                  # neighbors of the 5 last queries


###################################################################
# ------- GPU를 제대로 활용하고 있는지 테스트 ------

import numpy as np
import faiss
import time

# 데이터 생성
d = 128  # 차원
nb = 100000  # 데이터베이스 크기
nq = 10000  # 쿼리 크기
np.random.seed(1234)  # 재현성을 위한 시드 설정
xb = np.random.random((nb, d)).astype('float32')
xq = np.random.random((nq, d)).astype('float32')

# CPU 버전 테스트
cpu_index = faiss.IndexFlatL2(d)
cpu_index.add(xb)

start_time = time.time()
D, I = cpu_index.search(xq, k=5)  # 상위 5개 결과 검색
cpu_time = time.time() - start_time
print(f"CPU 검색 시간: {cpu_time:.4f} 초")

# GPU 버전 테스트
res = faiss.StandardGpuResources()  # GPU 리소스 초기화
gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

start_time = time.time()
D, I = gpu_index.search(xq, k=5)  # 상위 5개 결과 검색
gpu_time = time.time() - start_time
print(f"GPU 검색 시간: {gpu_time:.4f} 초")

# 속도 향상 계산
speedup = cpu_time / gpu_time
print(f"GPU 속도 향상: {speedup:.2f}배")