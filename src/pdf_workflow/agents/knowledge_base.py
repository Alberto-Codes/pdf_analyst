import ast
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, Dict, List

from agno.knowledge.langchain import LangChainKnowledgeBase
from langchain.docstore.document import Document
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_ollama import OllamaEmbeddings


def serialize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize complex metadata into a simpler format."""
    serialized = {}

    # Handle meta_data if present
    if "meta_data" in metadata and "dl_meta" in metadata["meta_data"]:
        try:
            # Parse dl_meta string to dict
            dl_meta = ast.literal_eval(metadata["meta_data"]["dl_meta"])

            # Extract document info
            serialized.update(
                {
                    "schema_name": dl_meta.get("schema_name"),
                    "version": dl_meta.get("version"),
                    "headings": dl_meta.get("headings", []),
                    "mimetype": dl_meta.get("origin", {}).get("mimetype"),
                    "binary_hash": dl_meta.get("origin", {}).get("binary_hash"),
                    "filename": dl_meta.get("origin", {}).get("filename"),
                }
            )

            # Extract text positions and bounding boxes
            text_positions = []
            for item in dl_meta.get("doc_items", []):
                for prov in item.get("prov", []):
                    position = {
                        "page_no": prov.get("page_no"),
                        "charspan": prov.get("charspan"),
                        "text_ref": item.get("self_ref"),
                    }
                    if "bbox" in prov:
                        bbox = prov["bbox"]
                        position["bbox"] = {
                            "left": bbox.get("l"),
                            "top": bbox.get("t"),
                            "right": bbox.get("r"),
                            "bottom": bbox.get("b"),
                            "coord_origin": bbox.get("coord_origin"),
                        }
                    text_positions.append(position)

            serialized["text_positions"] = text_positions

            # Keep source path
            if "source" in metadata["meta_data"]:
                serialized["source"] = metadata["meta_data"]["source"]
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing metadata: {e}")
            return metadata

    # Keep any other top-level metadata
    for key, value in metadata.items():
        if key != "meta_data" and isinstance(value, (str, int, float, bool)):
            serialized[key] = value

    return serialized


def process_documents(documents: List[Document]) -> List[Document]:
    """Process documents to serialize complex metadata."""
    processed_docs = []
    for doc in documents:
        doc.metadata = serialize_metadata(doc.metadata)
        processed_docs.append(doc)
    return processed_docs


def get_knowledge(file_path: Path) -> LangChainKnowledgeBase:
    """Load a knowledge base from a file.
    1. Load the document using DoclingLoader.
    2. Filter complex metadata from the documents.
    3. Store the filtered documents in a Chroma vector store.
    4. Create a retriever from the Chroma vector store.
    5. Return a LangChainKnowledgeBase object.

    Args:
        file_path (Path): The path to the document file.

    Returns:
        LangChainKnowledgeBase: The knowledge base object.
    """
    chroma_db_dir = mkdtemp()
    model = "nomic-embed-text"
    embeddings = OllamaEmbeddings(model=model)
    TOP_K = 10

    loader = DoclingLoader(file_path=file_path, export_type=ExportType.DOC_CHUNKS)
    documents = loader.load()

    # filtered_documents = filter_complex_metadata(documents)
    processed_documents = process_documents(documents)

    vector_db = Chroma.from_documents(
        # documents=filtered_documents,
        documents=processed_documents,
        embedding=embeddings,
        persist_directory=chroma_db_dir,
        collection_name="file_path",
    )

    # retriever = vector_db.as_retriever(search_type="mmr",
    #             search_kwargs={'k': 15, 'lambda_mult': 0.25})
    retriever = vector_db.as_retriever(search_kwargs={"k": 10})

    knowledge = LangChainKnowledgeBase(retriever=retriever)
    return knowledge
