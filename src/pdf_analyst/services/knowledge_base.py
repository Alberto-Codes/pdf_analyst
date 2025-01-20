from phi.knowledge.pdf import PDFUrlKnowledgeBase, PDFKnowledgeBase

from ..repositories.vector_db import VectorDbRepository


class KnowledgeBaseService:
    """Service to manage and load a PDF URL-based knowledge base.

    This class provides functionality to create and interact with a knowledge
    base sourced from a PDF URL. The knowledge base uses a vector database
    for storage and retrieval of data.
    """

    def __init__(self, pdf_source: str, is_url: bool = True):
        """Initializes the KnowledgeBaseService.

        Args:
            pdf_source (str): The URL or path of the PDF to be loaded into the knowledge base.
            is_url (bool, optional): Flag indicating whether the pdf_source is a URL. Defaults to True.
        """
        vector_db_repo = VectorDbRepository()
        if is_url:
            self.knowledge_base = PDFUrlKnowledgeBase(
                vector_db=vector_db_repo.vector_db,
                urls=[pdf_source],
            )
        else:
            self.knowledge_base = PDFKnowledgeBase(
                vector_db=vector_db_repo.vector_db,
                path="pdf_source",
            )

    def load_knowledge_base(self, recreate: bool = True):
        """Loads the knowledge base with the specified configuration.

        Args:
            recreate (bool, optional): Whether to recreate the knowledge base
                from scratch. Defaults to True.
        """
        self.knowledge_base.load(recreate=recreate)
