import os

from llama_index import (
    VectorStoreIndex,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)
from llama_index.llms import OpenAI
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.node_parser import SimpleNodeParser, SentenceWindowNodeParser
from llama_index.text_splitter import TokenTextSplitter
from llama_hub.confluence.base import ConfluenceReader

os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
os.environ["OPENAI_API_BASE"] = "http://localhost:8000/v1"

llm = OpenAI(temperature=0.1, max_tokens=2048)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

if os.path.exists("store"):
    index = load_index_from_storage(
        storage_context=StorageContext.from_defaults(persist_dir="store"),
        service_context=ServiceContext.from_defaults(llm=llm, embed_model=embed_model),
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

query_engine = index.as_query_engine(similarity_top_k=5, response_mode="no_text")

while 1:
    question = input("Question: ")
    answer = query_engine.query(question)
    for node in answer.source_nodes:
        print(f"{node.text} \n--------------------")
