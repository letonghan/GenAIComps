# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time

from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceHubEmbeddings
from langchain_community.vectorstores import Redis
from langsmith import traceable
from redis_config import EMBED_MODEL, INDEX_NAME, REDIS_URL
from langchain_core.prompts import ChatPromptTemplate
from comps import (
    EmbedDoc768,
    LLMParamsDoc,
    ServiceType,
    TextDoc,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)

tei_embedding_endpoint = os.getenv("TEI_EMBEDDING_ENDPOINT")


def ensure_length(s, desired_len=1024, fill_char=' '):
    if len(s) < desired_len:
        s += fill_char * (desired_len - len(s))
    elif len(s) > desired_len:
        s = s[:desired_len]
    return s


@register_microservice(
    name="opea_service@retriever_redis",
    service_type=ServiceType.RETRIEVER,
    endpoint="/v1/retrieval",
    host="0.0.0.0",
    port=7000,
)
@traceable(run_type="retriever")
@register_statistics(names=["opea_service@retriever_redis"])
def retrieve(input: EmbedDoc768) -> LLMParamsDoc:
    start = time.time()
    # check if the Redis index has data
    if vector_db.client.keys() == []:
        result = LLMParamsDoc(query=input.text)
        statistics_dict["opea_service@retriever_redis"].append_latency(time.time() - start, None)
        return result

    # if the Redis index has data, perform the search
    if input.search_type == "similarity":
        search_res = vector_db.similarity_search_by_vector(embedding=input.embedding, k=input.k)
    elif input.search_type == "similarity_distance_threshold":
        if input.distance_threshold is None:
            raise ValueError("distance_threshold must be provided for " + "similarity_distance_threshold retriever")
        search_res = vector_db.similarity_search_by_vector(
            embedding=input.embedding, k=input.k, distance_threshold=input.distance_threshold
        )
    elif input.search_type == "similarity_score_threshold":
        docs_and_similarities = vector_db.similarity_search_with_relevance_scores(
            query=input.text, k=input.k, score_threshold=input.score_threshold
        )
        search_res = [doc for doc, _ in docs_and_similarities]
    elif input.search_type == "mmr":
        search_res = vector_db.max_marginal_relevance_search(
            query=input.text, k=input.k, fetch_k=input.fetch_k, lambda_mult=input.lambda_mult
        )
    searched_docs = []
    for r in search_res:
        searched_docs.append(TextDoc(text=r.page_content))
    # template = """Answer the question based only on the following context:
    # {context}
    # Question: {question}
    # """
    # prompt = ChatPromptTemplate.from_template(template)
    doc = searched_docs[0]

    import re
    if re.findall("[\u4E00-\u9FFF]", doc.text) or re.findall("[\u4E00-\u9FFF]", input.text):
        # chinese context
        template = """
### 你将扮演一个乐于助人、尊重他人并诚实的助手，你的目标是帮助用户解答问题。有效地利用来自本地知识库的搜索结果。确保你的回答中只包含相关信息。如果你不确定问题的答案，请避免分享不准确的信息。
### 搜索结果：{context}
### 问题：{question}
### 回答：
"""
    else:
        template = """
### You are a helpful, respectful and honest assistant to help the user with questions. \
Please refer to the search results obtained from the local knowledge base. \
But be careful to not incorporate the information that you think is not relevant to the question. \
If you don't know the answer to a question, please don't share false information. \
### Search results: {context} \n
### Question: {question} \n
### Answer:
"""
    final_prompt = template.format(context=doc.text, question=input.text)
    final_prompt_1024 = ensure_length(final_prompt)
    result = LLMParamsDoc(query=final_prompt_1024)
    statistics_dict["opea_service@retriever_redis"].append_latency(time.time() - start, None)
    return result


if __name__ == "__main__":
    # Create vectorstore
    if tei_embedding_endpoint:
        # create embeddings using TEI endpoint service
        embeddings = HuggingFaceHubEmbeddings(model=tei_embedding_endpoint)
    else:
        # create embeddings using local embedding model
        embeddings = HuggingFaceBgeEmbeddings(model_name=EMBED_MODEL)

    vector_db = Redis(embedding=embeddings, index_name=INDEX_NAME, redis_url=REDIS_URL)
    opea_microservices["opea_service@retriever_redis"].start()
