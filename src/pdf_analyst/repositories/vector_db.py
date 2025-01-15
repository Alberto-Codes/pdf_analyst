from phi.vectordb.chroma import ChromaDb
from ..models.embedder import EmbedderModel

class VectorDbRepository:
    def __init__(self):
        embedder_model = EmbedderModel()
        self.vector_db = ChromaDb(collection="books", embedder=embedder_model.embedder, path="./tmp/chromadb")