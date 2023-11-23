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
from utils import ConfluenceReader, SentenceTransformerRerank

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
            project_key="MOBDATA",
            base_url="https://bitbucket.valtech.de",
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

    if os.path.exists(persist_dir):
        index = load_index_from_storage(
            storage_context=StorageContext.from_defaults(persist_dir="store"),
            service_context=ServiceContext.from_defaults(
                llm=LLM,
                embed_model=EMBEDDING,
                prompt_helper=PromptHelper(chunk_size_limit=2000),
            ),
        )
    else:
        index = VectorStoreIndex.from_documents(
            documents=ConfluenceReader(base_url=os.environ["CONFLUENCE_URL"]).load_data(
                space_key=os.environ["CONFLUENCE_SPACE"],
                page_status="current",
                include_attachments=False,
                max_num_results=10,
            ),
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
    return index


def get_query_engine(indices):
    config = {
        "similarity_top_k": 5,
        "response_mode": "compact",
        "node_postprocessors": [RERANK],
        "text_qa_template": PromptTemplate(
            "<|im_start|>system \n"
            "You will be presented with context. You task is to answer the query only based on the context. "
            "If the context cannot answer the query, you responses 'I don't know'. \n"
            "Approach this task step-by-step, take your time. \n"
            "This is very important to my career.<|im_end|>\n"
            "The Context information is below. \n"
            "---------------------\n{context_str}\n---------------------\n"
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
    conflunence_index = init_index(persist_dir="confluence_store")

    query_engine = get_query_engine(indices=[conflunence_index, bitbucket_index])
    print("[Develop mode]")
    while 1:
        question = input("Question: ")
        print(query_engine.query(question))
