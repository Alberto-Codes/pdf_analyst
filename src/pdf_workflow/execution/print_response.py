from dataclasses import dataclass

from config.state import GraphState
from pydantic_graph import BaseNode, End, GraphRunContext
from nodes.export import ExportToCSV


@dataclass
class PrintResponse(BaseNode[GraphState]):
    """Prints the response from the Gemini API.

    This class retrieves the response text from `GraphState` and prints it 
    to the console. It serves as a step in the graph execution to display 
    the generated content.

    Note:
        This class does not store attributes directly. The response text 
        is accessed from the `GraphState` in the `run` method.
    """

    async def run(self, ctx: GraphRunContext[GraphState]) -> ExportToCSV:
        """Prints the response text and returns the next node.

        This method prints the response text retrieved from `GraphState` 
        to the console and then returns an instance of `ExportToCSV`, 
        indicating the next step in the execution flow.

        Args:
            ctx (GraphRunContext[GraphState]): The execution context that 
                provides access to the shared state, including the response text.

        Returns:
            ExportToCSV: The next node in the graph execution for exporting 
            the response data to a CSV file.
        """
        # Print the generated response text to the console
        print(f"Gemini Response: {ctx.state.response_text}")

        # Return the next node in the execution flow
        return ExportToCSV()
