import os
import phoenix as px
import llama_index
from llama_index import (
    VectorStoreIndex,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)
from llama_index.llms import OpenAI
from llama_index.prompts import PromptTemplate
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.node_parser import SimpleNodeParser
from llama_index.text_splitter import TokenTextSplitter
from llama_index.indices.prompt_helper import PromptHelper
from llama_index.query_engine.router_query_engine import RouterQueryEngine
from llama_index.tools.query_engine import QueryEngineTool
from llama_index.selectors import EmbeddingSingleSelector 
from typing import Literal
from utils import ConfluenceReader, SentenceTransformerRerank, BitbucketReader

os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
os.environ["OPENAI_API_BASE"] = "http://localhost:8000/v1"

LLM = OpenAI(temperature=0.1, max_tokens=2048)
EMBEDDING = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
RERANK = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=3
)


def init_index(persist_dir: Literal["confluence_store", "bitbucket_store"]):
    if persist_dir == "bitbucket_store":
        loader = BitbucketReader(
            project_key=os.environ['BITBUCKET_PROJECT'],
            base_url=os.environ['BITBUCKET_URL'], 
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
        )
    elif persist_dir == "confluence_store":
        loader = ConfluenceReader(base_url=os.environ["CONFLUENCE_URL"]).load_data(
            space_key=os.environ["CONFLUENCE_SPACE"],
            page_status="current",
            include_attachments=False,
            max_num_results=10,
        )
    else:
        raise Exception("Must have one store")
    
    if os.path.exists(persist_dir):
        index = load_index_from_storage(
            storage_context=StorageContext.from_defaults(persist_dir=persist_dir),
            service_context=ServiceContext.from_defaults(
                llm=LLM,
                embed_model=EMBEDDING,
                prompt_helper=PromptHelper(chunk_size_limit=2000),
            ),
        )
    else:
        index = VectorStoreIndex.from_documents(
            documents=loader,
            service_context=ServiceContext.from_defaults(
                llm=LLM,
                node_parser=SimpleNodeParser.from_defaults(
                    text_splitter=TokenTextSplitter(
                        separator=" ",
                        chunk_size=512,
                        chunk_overlap=20,
                        backup_separators=["\n"],
                    )
                ),
                embed_model=EMBEDDING,
            ),
            show_progress=True,
        )
        index.storage_context.persist(persist_dir="store")
    
    index.__name__ = persist_dir
    return index


def get_query_engine(indices):
    config = {
        "similarity_top_k": 5,
        "response_mode": "compact",
        "node_postprocessors": [RERANK],
        "text_qa_template": PromptTemplate(
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
        ),
    }
    if len(indices) == 1:
        print("[Load Single Query Engine]")
        return indices[0].as_query_engine(
            similarity_top_k=config["similarity_top_k"],
            response_mode=config["response_mode"],
            node_postprocessors=config["node_postprocessors"],
            text_qa_template=config["text_qa_template"],
        )
    elif len(indices) > 1:
        engines = []
        for index in indices:
            engines.append(
                QueryEngineTool.from_defaults(
                    query_engine=index.as_query_engine(
                        similarity_top_k=config["similarity_top_k"],
                        response_mode=config["response_mode"],
                        node_postprocessors=config["node_postprocessors"],
                        text_qa_template=config["text_qa_template"],
                    ),
                    description=(f"Retrieve data from {index.__name__}"),
                )
            )
        return RouterQueryEngine(
            selector=EmbeddingSingleSelector.from_defaults(), query_engine_tools=engines
        )
    else:
        raise IndexError("It must contain one or more indices.")


if __name__ == "__main__":
    session = px.launch_app()
    llama_index.set_global_handler("arize_phoenix")

    bitbucket_index = init_index(persist_dir="bitbucket_store")
    confluence_index = init_index(persist_dir="confluence_store")

    query_engine_bitbucket = get_query_engine(indices=[bitbucket_index])
    # query_engine_confluence = get_query_engine(indices=[confluence_index])

    print("[Develop mode]")
    while 1:
        question = input("Question: ")
        print(query_engine_bitbucket.query(question))
        # print(query_engine_confluence.query(question))
        # response = embedding_compare(answer_from_bitbucket, answer_from_confluence, question)
