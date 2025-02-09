from config import GeminiConfig
from entities.officer import OFFICER_TEMPLATE
from graph import Graph
from graph_nodes import ExportNode, GenericExtractNode, GraphState, ParseNode


async def main():
    config = GeminiConfig()
    state = GraphState(pdf_path="data/10k.pdf", output_path="data/officers_export.csv")

    extract_node = GenericExtractNode(
        config=config,
        template=OFFICER_TEMPLATE,
        mime_type="application/pdf",
        stream_response=True,
        encoding="utf-8",
    )

    workflow = Graph(nodes={GenericExtractNode, ParseNode, ExportNode})
    result, history = await workflow.run(extract_node, state)

    print(f"Processed {len(result.entities)} entities")
    for step in history:
        print(f"Executed: {step.__class__.__name__}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
