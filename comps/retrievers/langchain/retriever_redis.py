# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceHubEmbeddings
from langchain_community.vectorstores import Redis
from langsmith import traceable
from redis_config import EMBED_MODEL, INDEX_NAME, INDEX_SCHEMA, REDIS_URL

from langchain_core.prompts import ChatPromptTemplate
from comps import EmbedDoc768, LLMParamsDoc, ServiceType, TextDoc, opea_microservices, register_microservice

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
def retrieve(input: EmbedDoc768) -> LLMParamsDoc:
    search_res = vector_db.similarity_search_by_vector(embedding=input.embedding)
    searched_docs = []
    for r in search_res:
        searched_docs.append(TextDoc(text=r.page_content))

    template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    doc = searched_docs[0]
    final_prompt = prompt.format(context=doc.text, question=input.text)
    final_prompt_1024 = ensure_length(final_prompt)
    return LLMParamsDoc(query=final_prompt_1024)


if __name__ == "__main__":
    # Create vectorstore
    if tei_embedding_endpoint:
        # create embeddings using TEI endpoint service
        embeddings = HuggingFaceHubEmbeddings(model=tei_embedding_endpoint)
    else:
        # create embeddings using local embedding model
        embeddings = HuggingFaceBgeEmbeddings(model_name=EMBED_MODEL)

    vector_db = Redis.from_existing_index(
        embedding=embeddings,
        index_name=INDEX_NAME,
        redis_url=REDIS_URL,
        schema=INDEX_SCHEMA,
    )
    opea_microservices["opea_service@retriever_redis"].start()
