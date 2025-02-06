FROM nvcr.io/nvidia/pytorch:24.12-py3

# 1. sudo 설치
RUN DEBIAN_FRONTEND=noninteractive apt update && \
    apt install -y sudo wget bzip2 && \
    rm -rf /var/lib/apt/lists/*

# 2. 사용자 및 그룹 생성
RUN groupadd -g 2101 oschung_skcc && \
    useradd -u 1002 -g 2101 -m oschung_skcc && \
    usermod -aG root oschung_skcc && \
    echo 'root:xx' | chpasswd && \
    echo 'oschung_skcc:xx' | chpasswd && \
    echo 'oschung_skcc ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# 3. Miniconda 설치
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# 4. Conda 경로 설정
ENV PATH /opt/conda/bin:$PATH
RUN chown -R oschung_skcc:oschung_skcc /opt/conda

# 5. Conda 초기화 및 사용자 환경 설정
USER oschung_skcc
RUN conda init bash && \
    echo "conda config --set auto_activate_base false" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    conda clean -ya
USER root

# 6. requirements.txt 복사 및 설치 스크립트 추가
# COPY requirements.txt /workspaces/
# RUN chown oschung_skcc:oschung_skcc /workspaces/requirements.txt

# 7. Conda 환경 업데이트 및 기본 패키지 설치
RUN conda update -n base -c defaults conda && \
    conda install -y python=3.12 jupyter ipykernel

# Jupyter 커널 등록 (시스템 전역 설치)
RUN python -m ipykernel install --name "python3.12" --display-name "Python 3.12.3"

# 작업 디렉토리 설정
WORKDIR /workspaces

# 기존 내용 하단에 추가
EXPOSE 8501 8888
