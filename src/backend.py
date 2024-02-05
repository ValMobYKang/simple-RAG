import os
from typing import Literal

import phoenix as px
from llama_index import (
    ServiceContext,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.callbacks import CallbackManager
from llama_index.core import BaseQueryEngine
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.indices.base import BaseIndex
from llama_index.indices.prompt_helper import PromptHelper
from llama_index.llms import OpenAI
from llama_index.postprocessor.types import BaseNodePostprocessor
from llama_index.prompts import PromptTemplate
from llama_index.query_engine import CustomQueryEngine
from llama_index.response_synthesizers import BaseSynthesizer, get_response_synthesizer
from llama_index.retrievers import BaseRetriever, BM25Retriever
from llama_index.schema import QueryBundle
from phoenix.trace.llama_index import OpenInferenceTraceCallbackHandler

from utils import BitbucketReader, ConfluenceReader, SentenceTransformerRerank

os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
os.environ["OPENAI_API_BASE"] = "http://localhost:8000/v1"

session = px.launch_app()
cb_manager = CallbackManager(handlers=[OpenInferenceTraceCallbackHandler()])


class QueryMultiEngine(CustomQueryEngine):
    retrievers: list[BaseRetriever]
    response_synthesizer: BaseSynthesizer
    node_postprocessors: list[BaseNodePostprocessor]

    def custom_query(self, query_str: str):
        nodes = []
        for retriever in self.retrievers:
            nodes += retriever.retrieve(query_str)
        for postprocessor in self.node_postprocessors:
            nodes = postprocessor.postprocess_nodes(
                nodes=nodes, query_bundle=QueryBundle(query_str)
            )

        return self.response_synthesizer.synthesize(query_str, nodes)


def service_context():
    LLM = OpenAI(
        temperature=0.1, max_tokens=2048, stop=["</s>"], callback_manager=cb_manager
    )
    EMBEDDING = HuggingFaceEmbedding(
        model_name="BAAI/bge-base-en-v1.5", callback_manager=cb_manager
    )
    return ServiceContext.from_defaults(
        llm=LLM,
        chunk_size=512,
        chunk_overlap=20,
        embed_model=EMBEDDING,
        prompt_helper=PromptHelper(chunk_size_limit=1000),
        callback_manager=cb_manager,
    )


def get_documents(
    input_dir: Literal["local_store", "confluence_store", "bitbucket_store"] = "local_store"
):
    documents = None
    if input_dir == "local_store":
        documents = SimpleDirectoryReader(
            input_dir=os.environ.get(
                "LOCAL_DIR", os.path.dirname(os.path.realpath(__file__))
            )
        ).load_data()
    elif input_dir == "bitbucket_store":
        documents = BitbucketReader(
            project_key=os.environ["BITBUCKET_PROJECT"],
            base_url=os.environ["BITBUCKET_URL"],
            branch="master",
            extensions_to_skip=[
                ".VIN-decoding",
                "URL-generalization",
                "scraping",
                "FizzBuzz",
                "Driver-Behaviour",
                "VIN-OCR",
                "Sensor-Log",
                "png",
                "jpg",
                "ppm",
            ],
        ).load_data()
    elif input_dir == "confluence_store":
        documents = ConfluenceReader(base_url=os.environ["CONFLUENCE_URL"]).load_data(
            space_key=os.environ["CONFLUENCE_SPACE"],
            page_status="current",
            include_attachments=False,
            max_num_results=10,
        )
    else:
        raise Exception("Store name should be 'local_store', 'bitbucket_store' or 'confluence_store' ")
    return documents


def init_index(persist_dir: str) -> BaseIndex:
    if os.path.exists(persist_dir):
        print(f"Found {persist_dir}, loading ...")
        return load_index_from_storage(
            storage_context=StorageContext.from_defaults(persist_dir=persist_dir),
            service_context=service_context(),
        )

    documents = get_documents(input_dir=persist_dir)

    index = VectorStoreIndex.from_documents(
        documents=documents,
        service_context=service_context(),
        show_progress=True,
    )
    index.storage_context.persist(persist_dir=persist_dir)

    return index


def get_bm25_retrievers(
    indices: list[BaseIndex], similarity_top_k: int = 5
) -> list[BaseRetriever]:
    retrievers = []
    for index in indices:
        retriever = BM25Retriever.from_defaults(
            index=index, similarity_top_k=similarity_top_k
        )
        retriever.callback_manager = cb_manager
        retrievers.append(retriever)
    return retrievers


def get_query_engine(indices: list[BaseIndex]) -> BaseQueryEngine:
    RERANK = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=3
    )
    RERANK.callback_manager = cb_manager
    dolphin_qa_prompt = PromptTemplate(
        "<|im_start|>system \n"
        "You will be presented with context. Your task is to answer the query only based on the context. "
        "If the context cannot answer the query, you responses 'I don't know' directly without any more responses. \n"
        "Approach this task step-by-step, take your time. "
        "This is very important to my career.\n"
        "The Context information is below. \n"
        "---------------------\n{context_str}\n--------------------- <|im_end|>\n"
        "<|im_start|>user \n"
        "{query_str}<|im_end|> \n"
        "<|im_start|>assistant"
    )

    mistral_qa_prompt = PromptTemplate(
        "<s>[INST] You will be presented with context. Your task is to answer the query only based on the context. "
        "If the context cannot answer the query, you responses 'I don't know' directly without any more responses. "
        "Approach this task step-by-step, take your time. This is very important to my career.\n"
        "The Context information is below. \n"
        "---------------------\n{context_str}\n--------------------- [/INST]</s>\n"
        "[INST] {query_str} [/INST]"
    )

    if len(indices) == 1:
        return indices[0].as_query_engine(
            similarity_top_k=5,
            service_context=service_context(),
            response_mode="compact",
            node_postprocessors=[RERANK],
            text_qa_template=mistral_qa_prompt,
        )

    bm25_retrievers = get_bm25_retrievers(indices)

    return QueryMultiEngine(
        retrievers=bm25_retrievers
        + [index.as_retriever(similarity_top_k=5) for index in indices],
        node_postprocessors=[RERANK],
        response_synthesizer=get_response_synthesizer(
            service_context=service_context(),
            response_mode="compact",
            text_qa_template=dolphin_qa_prompt,
        ),
        callback_manager=cb_manager,
    )


if __name__ == "__main__":
    print("[Develop mode]")
    query_engine_bitbucket = get_query_engine(
        indices=[
            init_index(persist_dir="bitbucket_store"),
            init_index(persist_dir="confluence_store"),
            init_index(persist_dir="local_store"),
        ]
    )

    while 1:
        question = input("Question: ")
        print(query_engine_bitbucket.query(question))
