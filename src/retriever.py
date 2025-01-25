import os
import faiss
from sentence_transformers import SentenceTransformer

class DocumentRetriever:
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.corpus = []
        self.metadata = []

    def preprocess_text(self, text):
        # Remove extra whitespaces, convert to lowercase
        text = ' '.join(text.split())
        return text.lower()

    def create_index(self, documents, metadata):
        processed_documents = [self.preprocess_text(doc) for doc in documents]
        embeddings = self.model.encode(processed_documents, show_progress_bar=True)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        print(f"Embeddings: {embeddings}")
        self.index.add(embeddings)
        self.corpus = documents
        self.metadata = metadata

    def search(self, query, top_k=5):
        # Preprocess query
        processed_query = self.preprocess_text(query)
        query_embedding = self.model.encode([processed_query])
        
        # Perform search with adjusted parameters
        distances, indices = self.index.search(query_embedding, top_k * 2)  # Retrieve more candidates
        
        # Re-rank results
        results = [
            {
                "text": self.corpus[i],
                "file": self.metadata[i],
                "distance": distances[0][j],
                "relevance_score": self.calculate_relevance(processed_query, self.corpus[i])
            }
            for j, i in enumerate(indices[0])
        ]
        
        # Sort by a combined score of distance and relevance
        results.sort(key=lambda x: (x['distance'], -x['relevance_score']))
        return results[:top_k]

    def calculate_relevance(self, query, document):
        # Implement more sophisticated relevance scoring
        # Could use techniques like TF-IDF, semantic similarity
        return len(set(query.split()) & set(document.split())) / len(set(query.split()))
        
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
