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
from llama_index.indices.prompt_helper import PromptHelper
from llama_index.query_engine import CustomQueryEngine
from llama_index.retrievers import BaseRetriever
from llama_index.response_synthesizers import (
    get_response_synthesizer,
    BaseSynthesizer,
)
from llama_index.postprocessor.types import BaseNodePostprocessor
from llama_index.schema import QueryBundle

from typing import Literal, List
from utils import ConfluenceReader, SentenceTransformerRerank , BitbucketReader

os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
os.environ["OPENAI_API_BASE"] = "http://localhost:8000/v1"

LLM = OpenAI(temperature=0.1, max_tokens=2048)
EMBEDDING = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
RERANK = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=3
)

class QueryEngine(CustomQueryEngine):
    retrievers: list[BaseRetriever]
    response_synthesizer: BaseSynthesizer
    node_postprocessors: list[BaseNodePostprocessor]

    def custom_query(self, query_str: str):

        nodes = []
        for retriever in self.retrievers:
            nodes += retriever.retrieve(query_str)
        print(len(nodes))
        for postprocessor in self.node_postprocessors:
            nodes = postprocessor.postprocess_nodes(nodes = nodes, query_bundle = QueryBundle(query_str))
        print(len(nodes))
 
        response_obj = self.response_synthesizer.synthesize(query_str, nodes)

        return response_obj
    

def get_service_context():
    return ServiceContext.from_defaults(
            llm=LLM,
            chunk_size=512,
            chunk_overlap=20,
            embed_model=EMBEDDING,
            prompt_helper=PromptHelper(chunk_size_limit=2000),
        )


def init_index(persist_dir: Literal["confluence_store", "bitbucket_store"]):

    if os.path.exists(persist_dir):
        return load_index_from_storage(
            storage_context=StorageContext.from_defaults(persist_dir=persist_dir),
            service_context=get_service_context(),
        )

    if persist_dir == "bitbucket_store":
        loader = BitbucketReader(
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
    elif persist_dir == "confluence_store":
        loader = ConfluenceReader(base_url=os.environ["CONFLUENCE_URL"]).load_data(
            space_key=os.environ["CONFLUENCE_SPACE"],
            page_status="current",
            include_attachments=False,
            max_num_results=10,
        )
    else:
        raise Exception("Must have one store")

    index = VectorStoreIndex.from_documents(
        documents=loader,
        service_context=get_service_context(),
        show_progress=True,
    )
    index.storage_context.persist(persist_dir=persist_dir)

    return index


def get_query_engine(indices):
    dolphin_prompt = PromptTemplate(
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

    query_engine = QueryEngine(
        retrievers=[index.as_retriever(similarity_top_k=5) for index in indices], 
        node_postprocessors=[RERANK],
         response_synthesizer=get_response_synthesizer(
            service_context=get_service_context(),
            response_mode="compact",        
            text_qa_template=dolphin_prompt
        )
    )

    return query_engine


if __name__ == "__main__":
    print("[Develop mode]")

    session = px.launch_app()
    llama_index.set_global_handler("arize_phoenix")

    bitbucket_index = init_index(persist_dir="bitbucket_store")
    confluence_index = init_index(persist_dir="confluence_store")

    query_engine_bitbucket = get_query_engine(indices=[bitbucket_index,confluence_index])

    while 1:
        question = input("Question: ")
        print(query_engine_bitbucket.query(question))
