from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vector_db = FAISS(embedding_function=embeddings)

def add_to_vector_db(text, metadata=None):
    vector_db.add_texts([text], metadatas=[metadata or {}])

def retrieve_from_vector_db(query, top_k=3):
    return vector_db.similarity_search(query, k=top_k)
