from config import GeminiConfig
from entities.officer import OFFICER_TEMPLATE
from graph import Graph
from graph_nodes import ExportNode, GenericExtractNode, GraphState, ParseNode


async def main():
    config = GeminiConfig()
    state = GraphState(pdf_path="data/10k.pdf", output_path="data/officers_export.csv")

    workflow = Graph(nodes={GenericExtractNode, ParseNode, ExportNode})
    result, history = await workflow.run(
        GenericExtractNode(config=config, template=OFFICER_TEMPLATE), state
    )

    print(f"Processed {len(result.entities)} officers")
    for step in history:
        print(f"Executed: {step.__class__.__name__}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
