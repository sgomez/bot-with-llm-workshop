import chromadb
from chromadb import Collection, QueryResult
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma

client = chromadb.PersistentClient(path="data/chroma.db")
collection: Collection = client.get_or_create_collection(name="reglamento-tfg-epsc")

embeddings_func = HuggingFaceEmbeddings()

database = Chroma(
    client=client,
    persist_directory="data/chroma.db",
    collection_name="reglamento-tfg-epsc",
    embedding_function=embeddings_func,
)


def predict(message: str) -> QueryResult:
    response = collection.query(query_texts=message, n_results=5)

    return response
