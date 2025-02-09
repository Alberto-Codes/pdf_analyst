from batch_processor import BatchProcessor
from config import GeminiConfig
from document_config import DocumentConfig
from entities.officer import OFFICER_TEMPLATE


async def main():
    config = GeminiConfig()
    doc_config = DocumentConfig(
        mime_type="application/pdf", stream_response=True, encoding="utf-8"
    )

    # Initialize batch processor
    processor = BatchProcessor(
        config=config,
        doc_config=doc_config,
        template=OFFICER_TEMPLATE,
        output_dir="data/officer_exports",
    )

    # List of PDFs to process
    documents = ["data/10k_2023.pdf", "data/10k_2022.pdf", "data/10k_2021.pdf"]

    # Process all documents
    results = await processor.process_documents(documents)

    # Print summary
    print("\nBatch Processing Summary:")
    print(f"Total documents processed: {len(results)}")
    total_entities = sum(len(result.entities) for result in results)
    print(f"Total entities extracted: {total_entities}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
