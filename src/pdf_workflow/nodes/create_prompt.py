from dataclasses import dataclass
from pathlib import Path
import logging
import os
from typing import Dict, Any, List

from pdf_workflow.config.state import GraphState
from google.genai import types
from pdf_workflow.nodes.execute_api import ExecuteAPI
from pdf_workflow.nodes.sanitize_prompt import SanitizePrompt
from pydantic_graph import BaseNode, GraphRunContext


@dataclass
class CreatePrompt(BaseNode[GraphState]):
    """Creates and prepares all content for the API request.

    This node is responsible for:
        1. Using the pre-encoded file content from EncodeFileNode
        2. Constructing the prompt
        3. Creating the full contents array with both document and prompt
        4. Storing everything in `GraphState` for subsequent nodes to use
    """

    async def run(self, ctx: GraphRunContext[GraphState]) -> SanitizePrompt:
        """Prepares all content for the API request.

        This method uses the pre-encoded document content (if available),
        constructs the prompt, and compiles the contents array. The contents
        are stored in the shared state (`GraphState`) for subsequent nodes to use.

        Args:
            ctx (GraphRunContext[GraphState]): The execution context containing
                the shared state.

        Returns:
            SanitizePrompt: The next node that will sanitize the prompt.
        """
        contents: List[types.Part] = []
        prompt = ctx.state.prompt

        # Use pre-encoded document if available
        if ctx.state.encoded_file and ctx.state.encoded_file.get("bytes"):
            logging.info("Using pre-encoded document content")
            contents.append(
                types.Part.from_bytes(
                    data=ctx.state.encoded_file["bytes"],
                    mime_type=ctx.state.encoded_file.get("mime_type", ctx.state.document_mime_type),
                )
            )
        elif ctx.state.document_path:
            # Fallback if EncodeFileNode somehow didn't encode the file
            logging.warning("Using fallback file encoding in CreatePrompt")
            filepath = Path(ctx.state.document_path)
            try:
                contents.append(
                    types.Part.from_bytes(
                        data=filepath.read_bytes(),
                        mime_type=ctx.state.document_mime_type,
                    )
                )
            except Exception as e:
                logging.error(f"Error reading file in CreatePrompt fallback: {str(e)}")
                # Continue with just the prompt

        # Add the prompt text
        contents.append(types.Part.from_text(text=prompt))

        ctx.state.contents = contents

        # Prepare Model Armor configuration
        model_armor_config: Dict[str, Any] = {
            "project_id": os.environ.get("MODEL_ARMOR_PROJECT_ID", "mock-project"),
            "location": ctx.state.location or "europe-west4",
            "template_id": os.environ.get("MODEL_ARMOR_TEMPLATE_ID")
        }

        # Return the next node for prompt sanitization with config
        return SanitizePrompt(config=model_armor_config)

