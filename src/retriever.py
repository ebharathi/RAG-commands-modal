import os
import faiss
from sentence_transformers import SentenceTransformer

class DocumentRetriever:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.corpus = []
        self.metadata = []

    def create_index(self, documents, metadata):
        embeddings = self.model.encode(documents, show_progress_bar=True)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        self.corpus = documents
        self.metadata = metadata

    def search(self, query, top_k=5):
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding, top_k)
        results = [{"text": self.corpus[i], "file": self.metadata[i], "distance": distances[0][j]} 
                   for j, i in enumerate(indices[0])]
        return results

def load_and_index_data(data_folder):
    from document_loader import load_document
    retriever = DocumentRetriever()
    documents = []
    metadata = []

    for file_name in os.listdir(data_folder):
        file_path = os.path.join(data_folder, file_name)
        try:
            content = load_document(file_path)
            documents.append(content)
            print(f"Loaded {file_name}[+]")
            metadata.append(file_name)
        except Exception as e:
            print(f"Error loading {file_name}: {e}")

    retriever.create_index(documents, metadata)
    return retriever
