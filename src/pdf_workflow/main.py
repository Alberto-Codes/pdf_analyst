from config import GeminiConfig
from graph import Graph
from graph_nodes import ExtractNode, GraphState, ParseNode, ExportNode

async def main():
    config = GeminiConfig()
    state = GraphState(
        pdf_path="data/10k.pdf",
        output_path="data/officers_export.csv"  # Optional: specify custom output path
    )
    
    workflow = Graph(nodes={ExtractNode, ParseNode, ExportNode})
    result, history = await workflow.run(ExtractNode(config), state)
    
    print(f"Processed {len(result.officers)} officers")
    for step in history:
        print(f"Executed: {step.__class__.__name__}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())