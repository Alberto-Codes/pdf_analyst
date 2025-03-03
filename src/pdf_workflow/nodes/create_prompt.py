from dataclasses import dataclass
from pathlib import Path

from pdf_workflow.config.state import GraphState
from google.genai import types
from pdf_workflow.nodes.execute_api import ExecuteAPI
from pydantic_graph import BaseNode, GraphRunContext


@dataclass
class CreatePrompt(BaseNode[GraphState]):
    """Creates and prepares all content for the API request.

    This node is responsible for:
        1. Reading and encoding the local document if provided.
        2. Constructing the prompt.
        3. Creating the full contents array with both document and prompt.
        4. Storing everything in `GraphState` for `ExecuteAPI` to use.
    """

    async def run(self, ctx: GraphRunContext[GraphState]) -> ExecuteAPI:
        """Prepares all content for the API request.

        This method reads the document from a local file path (if provided),
        constructs the prompt, and compiles the contents array. The contents
        are stored in the shared state (`GraphState`) for `ExecuteAPI` to use.

        Args:
            ctx (GraphRunContext[GraphState]): The execution context containing
                the shared state.

        Returns:
            ExecuteAPI: The next node that will execute the API request.
        """
        contents = []
        prompt = ctx.state.prompt

        # Handle document if a file path is provided
        if ctx.state.document_path:  # Changed from document_url
            filepath = Path(ctx.state.document_path)
            contents.append(
                types.Part.from_bytes(
                    data=filepath.read_bytes(),
                    mime_type=ctx.state.document_mime_type,
                )
            )

        contents.append(types.Part.from_text(text=prompt))

        ctx.state.contents = contents

        return ExecuteAPI()

