
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM python:3.11-slim

ENV LANG C.UTF-8

ARG ARCH="cpu"

RUN apt-get update -y && apt-get install -y --no-install-recommends --fix-missing \
    build-essential \
    libgl1-mesa-glx \
    libjemalloc-dev \
    vim

RUN useradd -m -s /bin/bash user && \
    mkdir -p /home/user && \
    chown -R user /home/user/

USER user

COPY comps /home/user/comps

RUN pip install --no-cache-dir --upgrade pip && \
    if [ ${ARCH} = "cpu" ]; then pip install torch --index-url https://download.pytorch.org/whl/cpu; fi && \
    pip install --no-cache-dir -r /home/user/comps/retrievers/langchain/milvus/requirements.txt

ENV PYTHONPATH=$PYTHONPATH:/home/user

USER root 

# Hard Code Langchain Milvus
RUN chmod +x /home/user/comps/retrievers/langchain/milvus/docker/hard_code.sh
RUN /home/user/comps/retrievers/langchain/milvus/docker/hard_code.sh

USER user

WORKDIR /home/user/comps/retrievers/langchain/milvus

# ENTRYPOINT ["python", "retriever_milvus.py"]

ENTRYPOINT ["sleep", "infinity"]

