from typing import List, Any
import streamlit as st

from llama_index.core import Document, SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.llms import CustomLLM
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def generate_engine_from_documents(doc_list: List[str], llm: CustomLLM,
                                   embedding_model: HuggingFaceEmbedding) -> BaseQueryEngine:
    """
    Function for processing uploaded files and generating query / chat index.
    Here we adopt sentence window method implemented in Llama-index.
    Args:
        settings:
        doc_list:
        llm:
        embedding_model:

    Returns:

    """
    """processing."""
    Settings.llm = llm
    Settings.embed_model = embedding_model
    documents = SimpleDirectoryReader(input_files=doc_list).load_data()
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text"
    )
    nodes = node_parser.get_nodes_from_documents(documents)
    sentence_index = VectorStoreIndex(nodes, embed_model=embedding_model)
    query_engine = sentence_index.as_query_engine(
        llm=llm,
        similarity_top_k=2,
        node_postprocessors=[
            MetadataReplacementPostProcessor(target_metadata_key="window")
        ],
    )
    return query_engine
