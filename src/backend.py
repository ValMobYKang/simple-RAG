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

# from llama_hub.confluence.base import ConfluenceReader
from utils import ConfluenceReader

os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
os.environ["OPENAI_API_BASE"] = "http://localhost:8000/v1"

llm = OpenAI(temperature=0.1, max_tokens=2048)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")


def init_index():
    if os.path.exists("store"):
        index = load_index_from_storage(
            storage_context=StorageContext.from_defaults(persist_dir="store"),
            service_context=ServiceContext.from_defaults(
                llm=llm, embed_model=embed_model
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
                llm=llm,
                node_parser=SimpleNodeParser.from_defaults(
                    text_splitter=TokenTextSplitter(
                        separator=" ",
                        chunk_size=512,
                        chunk_overlap=20,
                        backup_separators=["\n"],
                    )
                ),
                embed_model=embed_model,
            ),
            show_progress=True,
        )
        index.storage_context.persist(persist_dir="store")
    return index


def get_query_engine(index):
    return index.as_query_engine(
        similarity_top_k=2,
        response_mode="compact",
        text_qa_template=PromptTemplate(
            "<|im_start|>system \n"
            "Given the context information and no prior knowledge, answer the query. If you dont know the answer, reply 'I dont know!' without any further content. This is very important to my career.<|im_end|> \n"
            "Context information is below. \n"
            "---------------------\n{context_str}\n---------------------\n"
            "<|im_start|>user \n"
            "{query_str}<|im_end|> \n"
            "<|im_start|>assistant"
        ),
    )


if __name__ == "__main__":
    session = px.launch_app()
    llama_index.set_global_handler("arize_phoenix")

    query_engine = get_query_engine(index=init_index())
    print("[Develop mode]")
    while 1:
        question = input("Question: ")
        print(query_engine.query(question))
