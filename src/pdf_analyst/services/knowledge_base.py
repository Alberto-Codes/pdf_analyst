from phi.knowledge.pdf import PDFUrlKnowledgeBase
from ..repositories.vector_db import VectorDbRepository

class KnowledgeBaseService:
    def __init__(self):
        vector_db_repo = VectorDbRepository()
        self.knowledge_base = PDFUrlKnowledgeBase(
            urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
            vector_db=vector_db_repo.vector_db,
        )

    def load_knowledge_base(self, recreate: bool = True):
        self.knowledge_base.load(recreate=recreate)