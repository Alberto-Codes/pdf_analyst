from typing import Dict, Generic, List, Tuple, Type, TypeVar

from nodes.extract import BaseNode, End, GraphState

T = TypeVar("T")


class Graph(Generic[T]):
    """Simple graph implementation for workflow execution."""

    def __init__(self, nodes: Dict[str, Type[BaseNode[GraphState]]]):
        """Nodes should be passed as a dictionary with explicit names."""
        self.nodes = nodes

    def get_node(self, node_name: str) -> Type[BaseNode[GraphState]]:
        """Retrieve a node by its name, enforcing structured workflow."""
        if node_name not in self.nodes:
            raise ValueError(f"Node '{node_name}' not found in the workflow.")
        return self.nodes[node_name]

    async def run(
        self, start_node: BaseNode[GraphState], state: GraphState
    ) -> Tuple[T, List[BaseNode[GraphState]]]:
        history = []
        current_node = start_node

        while True:
            history.append(current_node)
            result = await current_node.run(state)

            if isinstance(result, End):
                return result.data, history

            current_node = result
