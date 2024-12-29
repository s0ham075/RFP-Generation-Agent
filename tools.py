from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
)
import re
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.astra_db import AstraDBVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.tools import FunctionTool
from typing import List
from llama_index.core.schema import NodeWithScore
from typing import Optional
load_dotenv()
# api_endpoint = "https://eef12881-687b-44e6-80ab-75d09b40338d-us-east-2.apps.astra.datastax.com"
# token = "AstraCS:JKzdpnyarsePNAopwJgOWMyC:703b964d04605e8871fa1b73ab0d531ca7eef918cd1e3447c851ac6c8f32bd3d"
# collection_name = "kb_aiplanet"
# embedding_dimension = 768
# collection_description = "A sample collection containing RFP-related data."

api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
token = os.getenv("ASTRA_DB_TOKEN")
collection_name = os.getenv("ASTRA_DB_COLLECTION_NAME")
embedding_dimension = int(os.getenv("EMBEDDING_DIMENSION", "768"))
collection_description = os.getenv("COLLECTION_DESCRIPTION", "Default description.")



def get_all_documents(vector_store: AstraDBVectorStore) -> list:
    """
    Retrieve all documents from an AstraDB collection using the vector store's client.
    
    Args:
        vector_store (AstraDBVectorStore): An initialized AstraDB vector store
        
    Returns:
        list: List of all documents in the collection
    """
    # Access the underlying astrapy Collection object
    collection = vector_store.client
    
    documents = list(collection.find(
        filter={},  # No filters
        projection={"*": True},  # Get all fields
        limit=10000  # Adjust this based on your expected collection size
    ))
    
    return documents



def generate_tool(file,index):
    """Return a function that retrieves only within a given file."""

    def chunk_retriever_fn(query: str) -> str:
        retriever = index.as_retriever(similarity_top_k=5)
        nodes = retriever.retrieve(query)

        full_text = "\n\n========================\n\n".join(
            [n.get_content(metadata_mode="all") for n in nodes]
        )

        return full_text

    id = file["_id"].split("-")[0]
    file_name_sanitized = re.sub(r"[^a-zA-Z0-9_-]", "", file["metadata"]["file_name"])
    fn_name = f"{file_name_sanitized}{id}_retrieve"

    tool_description = f"Retrieves a small set of relevant document chunks from {file["metadata"]["file_name"]}."
    if file["metadata"]["summary"] is not None:
        tool_description += f"\n\nFile Description: {file["metadata"]["summary"]}"

    tool = FunctionTool.from_defaults(
        fn=chunk_retriever_fn, name=fn_name, description=tool_description
    )

    return tool


def generate_tools():
    astra_db_store = AstraDBVectorStore(
    token=token,
    api_endpoint=api_endpoint,
    collection_name=collection_name,
    embedding_dimension=embedding_dimension,
    )

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    index = VectorStoreIndex.from_vector_store(vector_store=astra_db_store, embed_model=embed_model)

    docs = get_all_documents(astra_db_store)
    tools = []
    for doc in docs:
        tools.append(generate_tool(doc,index))

    return tools

if __name__ == "__main__":
    tools = generate_tools()
    print(tools[0].metadata)