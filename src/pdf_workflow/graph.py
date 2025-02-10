from typing import List, Tuple, Type, TypeVar

from nodes.extract import BaseNode, End, GraphState
from pydantic_graph import Graph as PydanticGraph
from pydantic_graph import GraphRunContext

T = TypeVar("T")


class Graph(PydanticGraph[GraphState, None, T]):
    """Graph implementation using pydantic-graph for structured workflow execution."""

    def __init__(self, nodes: Tuple[Type[BaseNode], ...]):
        """Nodes should be passed as a tuple of node classes."""
        self.nodes = nodes

    async def run(
        self, start_node: BaseNode, state: GraphState
    ) -> Tuple[T, List[BaseNode]]:
        """Run the graph with a wrapped GraphRunContext."""
        ctx = GraphRunContext(state=state, deps=None)
        history = []
        current_node = start_node

        while True:
            history.append(current_node)
            result = await current_node.run(ctx)

            if isinstance(result, End):
                return result.data, history

            current_node = result
