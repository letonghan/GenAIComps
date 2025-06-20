# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Union

import numpy as np
from docarray import BaseDoc, DocList
from docarray.documents import AudioDoc
from docarray.typing import AudioUrl, ImageUrl
from pydantic import Field, NonNegativeFloat, PositiveInt, conint, conlist, field_validator


class TopologyInfo:
    # will not keep forwarding to the downstream nodes in the black list
    # should be a pattern string
    downstream_black_list: Optional[list] = []


class TextDoc(BaseDoc, TopologyInfo):
    text: Union[str, List[str]] = None


class Audio2text(BaseDoc, TopologyInfo):
    query: str = None


class FactualityDoc(BaseDoc):
    reference: str
    text: str


class ScoreDoc(BaseDoc):
    score: float


class PIIRequestDoc(BaseDoc):
    prompt: str
    replace: Optional[bool] = False
    replace_method: Optional[str] = "random"


class PIIResponseDoc(BaseDoc):
    detected_pii: Optional[List[dict]] = None
    new_prompt: Optional[str] = None


class MetadataTextDoc(TextDoc):
    metadata: Optional[Dict[str, Any]] = Field(
        description="This encloses all metadata associated with the textdoc.",
        default=None,
    )


class ImageDoc(BaseDoc):
    url: Optional[ImageUrl] = Field(
        description="The path to the image. It can be remote (Web) URL, or a local file path",
        default=None,
    )
    base64_image: Optional[str] = Field(
        description="The base64-based encoding of the image",
        default=None,
    )


class TextImageDoc(BaseDoc):
    image: ImageDoc = None
    text: TextDoc = None


MultimodalDoc = Union[
    TextDoc,
    ImageDoc,
    TextImageDoc,
]


class Base64ByteStrDoc(BaseDoc):
    byte_str: str


class DocSumDoc(BaseDoc):
    text: Optional[str] = None
    audio: Optional[str] = None
    video: Optional[str] = None


class DocPath(BaseDoc):
    path: str
    chunk_size: int = 1500
    chunk_overlap: int = 100
    process_table: bool = False
    table_strategy: str = "fast"


class EmbedDoc(BaseDoc):
    text: Union[str, List[str]]
    embedding: Union[conlist(float, min_length=0), List[conlist(float, min_length=0)]]
    search_type: str = "similarity"
    k: PositiveInt = 4
    distance_threshold: Optional[float] = None
    fetch_k: PositiveInt = 20
    lambda_mult: NonNegativeFloat = 0.5
    score_threshold: NonNegativeFloat = 0.2
    constraints: Optional[Union[Dict[str, Any], List[Dict[str, Any]], None]] = None
    index_name: Optional[str] = None


class EmbedMultimodalDoc(EmbedDoc):
    # extend EmbedDoc with these attributes
    url: Optional[ImageUrl] = Field(
        description="The path to the image. It can be remote (Web) URL, or a local file path.",
        default=None,
    )
    base64_image: Optional[str] = Field(
        description="The base64-based encoding of the image.",
        default=None,
    )


class Audio2TextDoc(AudioDoc):
    url: Optional[AudioUrl] = Field(
        description="The path to the audio.",
        default=None,
    )
    model_name_or_path: Optional[str] = Field(
        description="The Whisper model name or path.",
        default="openai/whisper-small",
    )
    language: Optional[str] = Field(
        description="The language that Whisper prefer to detect.",
        default="auto",
    )


class SearchedDoc(BaseDoc):
    retrieved_docs: DocList[TextDoc]
    initial_query: str
    top_n: PositiveInt = 1

    class Config:
        json_encoders = {np.ndarray: lambda x: x.tolist()}


class SearchedMultimodalDoc(SearchedDoc):
    metadata: List[Dict[str, Any]]


class LVMSearchedMultimodalDoc(SearchedMultimodalDoc):
    max_new_tokens: conint(ge=0, le=1024) = 512
    top_k: int = 10
    top_p: float = 0.95
    typical_p: float = 0.95
    temperature: float = 0.01
    stream: bool = False
    repetition_penalty: float = 1.03
    chat_template: Optional[str] = Field(
        default=None,
        description=(
            "A template to use for this conversion. "
            "If this is not passed, the model's default chat template will be "
            "used instead. We recommend that the template contains {context} and {question} for multimodal-rag on videos."
        ),
    )


class GeneratedDoc(BaseDoc):
    text: str
    prompt: str


class RerankedDoc(BaseDoc):
    reranked_docs: DocList[TextDoc]
    initial_query: str


class LLMParamsDoc(BaseDoc):
    model: Optional[str] = None  # for openai and ollama
    query: str
    max_tokens: int = 1024
    max_new_tokens: PositiveInt = 1024
    top_k: PositiveInt = 10
    top_p: NonNegativeFloat = 0.95
    typical_p: NonNegativeFloat = 0.95
    temperature: NonNegativeFloat = 0.01
    frequency_penalty: NonNegativeFloat = 0.0
    presence_penalty: NonNegativeFloat = 0.0
    repetition_penalty: NonNegativeFloat = 1.03
    stream: bool = True
    language: str = "auto"  # can be "en", "zh"

    chat_template: Optional[str] = Field(
        default=None,
        description=(
            "A template to use for this conversion. "
            "If this is not passed, the model's default chat template will be "
            "used instead. We recommend that the template contains {context} and {question} for rag,"
            "or only contains {question} for chat completion without rag."
        ),
    )
    documents: Optional[Union[List[Dict[str, str]], List[str]]] = Field(
        default=[],
        description=(
            "A list of dicts representing documents that will be accessible to "
            "the model if it is performing RAG (retrieval-augmented generation)."
            " If the template does not support RAG, this argument will have no "
            "effect. We recommend that each document should be a dict containing "
            '"title" and "text" keys.'
        ),
    )

    @field_validator("chat_template")
    def chat_template_must_contain_variables(cls, v):
        return v


class LLMParams(BaseDoc):
    model: Optional[str] = None
    max_tokens: int = 1024
    max_new_tokens: PositiveInt = 1024
    top_k: PositiveInt = 10
    top_p: NonNegativeFloat = 0.95
    typical_p: NonNegativeFloat = 0.95
    temperature: NonNegativeFloat = 0.01
    frequency_penalty: NonNegativeFloat = 0.0
    presence_penalty: NonNegativeFloat = 0.0
    repetition_penalty: NonNegativeFloat = 1.03
    stream: bool = True
    language: str = "auto"  # can be "en", "zh"
    index_name: Optional[str] = None

    chat_template: Optional[str] = Field(
        default=None,
        description=(
            "A template to use for this conversion. "
            "If this is not passed, the model's default chat template will be "
            "used instead. We recommend that the template contains {context} and {question} for rag,"
            "or only contains {question} for chat completion without rag."
        ),
    )


class RetrieverParms(BaseDoc):
    search_type: str = "similarity"
    k: PositiveInt = 4
    distance_threshold: Optional[float] = None
    fetch_k: PositiveInt = 20
    lambda_mult: NonNegativeFloat = 0.5
    score_threshold: NonNegativeFloat = 0.2


class RerankerParms(BaseDoc):
    top_n: PositiveInt = 1


class RAGASParams(BaseDoc):
    questions: DocList[TextDoc]
    answers: DocList[TextDoc]
    docs: DocList[TextDoc]
    ground_truths: DocList[TextDoc]


class RAGASScores(BaseDoc):
    answer_relevancy: float
    faithfulness: float
    context_recallL: float
    context_precision: float


class GraphDoc(BaseDoc):
    text: str
    strtype: Optional[str] = Field(
        description="type of input query, can be 'query', 'cypher', 'rag'",
        default="query",
    )
    max_new_tokens: Optional[int] = Field(default=1024)
    rag_index_name: Optional[str] = Field(default="rag")
    rag_node_label: Optional[str] = Field(default="Task")
    rag_text_node_properties: Optional[list] = Field(default=["name", "description", "status"])
    rag_embedding_node_property: Optional[str] = Field(default="embedding")


class LVMDoc(BaseDoc):
    image: Union[str, List[str]]
    prompt: str
    max_new_tokens: conint(ge=0, le=1024) = 512
    top_k: int = 10
    top_p: float = 0.95
    typical_p: float = 0.95
    temperature: float = 0.01
    repetition_penalty: float = 1.03
    stream: bool = False


class LVMVideoDoc(BaseDoc):
    video_url: str
    chunk_start: float
    chunk_duration: float
    prompt: str
    max_new_tokens: conint(ge=0, le=1024) = 512


class SDInputs(BaseDoc):
    prompt: str
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    num_images_per_prompt: int = 1
    seed: int = 42
    negative_prompt: Optional[Union[str, List[str]]] = None
    height: Optional[int] = None
    width: Optional[int] = None
    lora_weight_name_or_path: Optional[str] = None


class SDImg2ImgInputs(BaseDoc):
    image: str
    prompt: Union[str, List[str]] = None
    height: Optional[int] = None
    width: Optional[int] = None
    strength: float = 0.8
    num_inference_steps: Optional[int] = 50
    guidance_scale: Optional[float] = 7.5
    negative_prompt: Optional[Union[str, List[str]]] = None
    num_images_per_prompt: int = 1
    seed: int = 42
    lora_weight_name_or_path: Optional[str] = None


class SDOutputs(BaseDoc):
    images: list


class ImagePath(BaseDoc):
    image_path: str


class ImagesPath(BaseDoc):
    images_path: DocList[ImagePath]


class VideoPath(BaseDoc):
    video_path: str


class PrevQuestionDetails(BaseDoc):
    question: str
    answer: str


class PromptTemplateInput(BaseDoc):
    data: Dict[str, Any]
    conversation_history: Optional[List[PrevQuestionDetails]] = None
    conversation_history_parse_type: str = "naive"
    system_prompt_template: Optional[str] = None
    user_prompt_template: Optional[str] = None


class TranslationInput(BaseDoc):
    text: str
    target_language: str
