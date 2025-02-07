FROM nvcr.io/nvidia/pytorch:24.12-py3
# 기본 패키지 설치
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    sudo wget bzip2 bash-completion curl vim nano net-tools htop && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
# 사용자 추가 및 권한 설정
ARG USERNAME=oschung_skcc
ARG USER_UID=1002
ARG USER_GID=2101
RUN groupadd --gid ${USER_GID} ${USERNAME} && \
    useradd --uid ${USER_UID} --gid ${USER_GID}  -m ${USERNAME} && \
    echo ${USERNAME} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USERNAME} && \
    chmod 0440 /etc/sudoers.d/${USERNAME} && \
    echo 'root:xx' | chpasswd && \
    echo 'oschung_skcc:xx' | chpasswd
# Python 가상 환경 생성
USER ${USERNAME}
RUN python -m venv /home/${USERNAME}/venv
ENV PATH="/home/${USERNAME}/venv/bin:$PATH"
# 4. 기본 패키지 설치
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install jupyter ipykernel
# 5. 작업 디렉토리 설정
WORKDIR /workspaces
# 6. 포트 설정
EXPOSE 8501 8888
