import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

class FAISSIndex:
    def __init__(self, faiss_index, metadata):
        self.index = faiss_index
        self.metadata = metadata

    def similarity_search(self, query, k=3):
        query = np.array([query]).astype("float32")
        D, I = self.index.search(query, k)
        results = []
        for idx in I[0]:
            results.append(self.metadata[idx])
        return results

embed_model_id = "sentence-transformers/all-MiniLM-L6-v2"  # nazwa modelu
model_kwargs = {"device": "cpu", "trust_remote_code": True}

def create_index(documents):
    embeddings = HuggingFaceEmbeddings(model_name=embed_model_id, model_kwargs=model_kwargs)
    texts = [doc.page_content for doc in documents]  # wartości tekstowe wszystkich dokumentów
    metadata = [{"filename": doc.metadata.get("filename"), "text": doc.page_content} for doc in documents]  # metadane wszystkich dokumentów, czyli słownik {filename:... , text:...}

    embeddings_matrix = [embeddings.embed_query(text) for text in texts]
    embeddings_matrix = np.array(embeddings_matrix).astype("float32")

    index = faiss.IndexFlatIP(len(embeddings_matrix[0]))
    index.add(embeddings_matrix)

    return FAISSIndex(index, metadata)

def retrieve_docs(query, faiss_index, k=3):
    embeddings = HuggingFaceEmbeddings(model_name=embed_model_id, model_kwargs=model_kwargs)
    query_embedding = embeddings.embed_query(query)
    results = faiss_index.similarity_search(query_embedding, k=k)
    return results