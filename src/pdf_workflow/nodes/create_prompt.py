import tempfile
import urllib.request
from dataclasses import dataclass

from config.state import GraphState
from google.genai import types
from nodes.execute_api import ExecuteAPI
from pydantic_graph import BaseNode, GraphRunContext


@dataclass
class CreatePrompt(BaseNode[GraphState]):
    """Creates and prepares all content for the API request.

    This node is responsible for:
        1. Downloading and encoding the document if provided.
        2. Constructing the prompt.
        3. Creating the full contents array with both document and prompt.
        4. Storing everything in `GraphState` for `ExecuteAPI` to use.
    """

    async def run(self, ctx: GraphRunContext[GraphState]) -> ExecuteAPI:
        """Prepares all content for the API request.

        This method downloads and encodes the document (if a URL is provided),
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

        # Handle document if a URL is provided
        if ctx.state.document_url:
            with tempfile.NamedTemporaryFile(mode="wb+", delete=True) as temp_file:
                # Download file content
                with urllib.request.urlopen(ctx.state.document_url) as response:
                    temp_file.write(response.read())

                # Reset file pointer and read file as bytes
                temp_file.seek(0)
                file_bytes = temp_file.read()

                contents.append(
                    types.Part.from_bytes(
                        data=file_bytes,
                        mime_type=ctx.state.document_mime_type,
                    )
                )

        contents.append(types.Part.from_text(text=prompt))

        ctx.state.contents = contents

        return ExecuteAPI()
