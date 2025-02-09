from typing import Generic, List, Set, Tuple, Type, TypeVar

from graph_nodes import BaseNode, End, GraphState

T = TypeVar("T")


class Graph(Generic[T]):
    """Simple graph implementation for workflow execution."""

    def __init__(self, nodes: Set[Type[BaseNode[GraphState]]]):
        self.nodes = nodes

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
