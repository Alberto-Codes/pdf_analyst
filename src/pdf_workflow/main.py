import asyncio

from batch_processor import BatchProcessor
from config import GeminiConfig
from document_config import DocumentConfig
from entities.employee import EMPLOYEE_TEMPLATE
from entities.officer import OFFICER_TEMPLATE


async def main():
    config = GeminiConfig()
    doc_config = DocumentConfig(
        mime_type="application/pdf", stream_response=True, encoding="utf-8"
    )

    # Process officers
    officer_processor = BatchProcessor(
        config=config,
        doc_config=doc_config,
        template=OFFICER_TEMPLATE,
        output_dir="data/officer_exports",
        max_concurrent=3,
    )

    # Process employee counts
    employee_processor = BatchProcessor(
        config=config,
        doc_config=doc_config,
        template=EMPLOYEE_TEMPLATE,
        output_dir="data/employee_exports",
        max_concurrent=3,
    )

    documents = [
        "data/10k_2023.pdf",
        "data/10k_2022.pdf",
    ]

    # Process both entity types
    print("\nProcessing Officers...")
    officer_results = await officer_processor.process_documents(documents)

    print("\nProcessing Employee Counts...")
    employee_results = await employee_processor.process_documents(documents)

    # Print combined summary
    print("\nFinal Processing Summary:")
    print(f"Documents processed: {len(documents)}")
    print(f"Total officers found: {sum(len(r.entities) for r in officer_results)}")
    print(f"Employee counts extracted: {len(employee_results)}")


if __name__ == "__main__":
    asyncio.run(main())
