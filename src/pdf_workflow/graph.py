from __future__ import annotations

from typing import Any, List, Tuple, Type, TypeVar

from nodes.base import GraphState
from nodes.extract import BaseNode, End
from pydantic import BaseModel
from pydantic_graph import Graph as PydanticGraph
from pydantic_graph import GraphRunContext

T = TypeVar("T")
D = TypeVar("D")  # New type variable for dependencies


class Graph(PydanticGraph[GraphState, D, T]):
    """Graph implementation using pydantic-graph for structured workflow execution."""

    def __init__(self, nodes: Tuple[Type[BaseNode], ...]):
        """Initialize graph with node types."""
        self.nodes = nodes

    async def run(
        self,
        start_node: BaseNode,
        state: GraphState,
        deps: D | None = None,
    ) -> Tuple[T, List[BaseNode]]:
        """Run the graph with a wrapped GraphRunContext."""
        ctx = GraphRunContext(state=state, deps=deps)
        history = []
        current_node = start_node

        while True:
            history.append(current_node)
            result = await current_node.run(ctx)

            if isinstance(result, End):
                return result.data, history

            current_node = result

    def save_workflow_diagram(self, output_path: str, highlighted_nodes: List[Type[BaseNode]] | None = None) -> None:
        """Generate and save a mermaid diagram of the workflow.
        
        Args:
            output_path: Path to save the diagram PNG file
            highlighted_nodes: Optional list of node types to highlight in the diagram
        """
        self.mermaid_save(output_path, highlighted_nodes=highlighted_nodes)
