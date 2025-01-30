from __future__ import annotations

from datetime import datetime
from typing import AsyncIterator
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator as AsyncIteratorType

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference as WatsonModelInference

from ..models import Model, AgentModel, StreamedResponse, ModelMessage, ModelResponse, Usage, ModelSettings

class WatsonXModel(Model):
    """Implementation of IBM WatsonX.ai model for pydantic-ai."""
    
    def __init__(
        self,
        model_id: str,
        credentials: dict | Credentials,
        project_id: str | None = None,
        space_id: str | None = None,
    ):
        """Initialize WatsonX.ai model.
        
        Args:
            model_id: The WatsonX.ai model ID to use
            credentials: WatsonX.ai credentials
            project_id: Optional project ID 
            space_id: Optional space ID
        """
        self.model_id = model_id
        self.credentials = credentials
        self.project_id = project_id
        self.space_id = space_id
        
        # Initialize the WatsonX.ai model inference
        self.model = WatsonModelInference(
            model_id=model_id,
            credentials=credentials,
            project_id=project_id,
            space_id=space_id
        )

    def name(self) -> str:
        """Get the name of the model."""
        return f"watsonx:{self.model_id}"

    async def agent_model(
        self,
        *,
        function_tools: list,  # ToolDefinition
        allow_text_result: bool,
        result_tools: list,  # ToolDefinition
    ) -> AgentModel:
        """Create an agent model instance."""
        return WatsonXAgentModel(
            model=self.model,
            function_tools=function_tools,
            allow_text_result=allow_text_result,
            result_tools=result_tools
        )

class WatsonXAgentModel(AgentModel):
    """Implementation of IBM WatsonX.ai agent model."""
    
    def __init__(
        self,
        model: WatsonModelInference,
        function_tools: list,
        allow_text_result: bool,
        result_tools: list,
    ):
        self.model = model
        self.function_tools = function_tools
        self.allow_text_result = allow_text_result
        self.result_tools = result_tools

    def _convert_message_to_watson_format(self, msg: ModelMessage) -> dict:
        """Convert a pydantic-ai message to WatsonX.ai format."""
        # Default to 'user' if no role specified
        role = 'user'
        
        # Map message types to roles
        if msg.type == 'system':
            role = 'system'
        elif msg.type == 'assistant':
            role = 'assistant'
        elif msg.type == 'user':
            role = 'user'
        elif msg.type == 'tool':
            # Handle tool messages - might need adjustment based on WatsonX.ai's expectations
            role = 'assistant'
            
        return {
            "role": role,
            "content": msg.content if isinstance(msg.content, str) else str(msg.content)
        }

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None = None,
    ) -> tuple[ModelResponse, Usage]:
        """Make a request to the WatsonX.ai model."""
        # Convert pydantic-ai messages to WatsonX.ai format
        watson_messages = [
            self._convert_message_to_watson_format(msg)
            for msg in messages
        ]
        
        # Make the request to WatsonX.ai
        response = await self.model.achat(messages=watson_messages)
        
        # Convert WatsonX.ai response to pydantic-ai format
        model_response = ModelResponse(
            parts=[response["choices"][0]["message"]["content"]],
            model_name=self.model.model_id,
            timestamp=datetime.now()
        )
        
        # Extract usage information
        usage = Usage(
            prompt_tokens=response.get("usage", {}).get("prompt_tokens", 0),
            completion_tokens=response.get("usage", {}).get("completion_tokens", 0),
            total_tokens=response.get("usage", {}).get("total_tokens", 0)
        )
        
        return model_response, usage

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        """Make a streaming request to the WatsonX.ai model."""
        # Convert messages to WatsonX.ai format
        watson_messages = [
            self._convert_message_to_watson_format(msg)
            for msg in messages
        ]
        
        # Create streaming response
        stream = WatsonXStreamedResponse(
            model=self.model,
            messages=watson_messages,
            model_name=self.model.model_id
        )
        
        try:
            yield stream
        finally:
            # Cleanup if needed
            pass

class WatsonXStreamedResponse(StreamedResponse):
    """Implementation of IBM WatsonX.ai streamed response."""
    
    def __init__(
        self,
        model: WatsonModelInference,
        messages: list[dict],
        model_name: str
    ):
        super().__init__(model_name=model_name)
        self.model = model
        self.messages = messages
        self._timestamp = datetime.now()

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        """Get async iterator for stream events."""
        # Use WatsonX.ai chat_stream method
        stream = self.model.chat_stream(messages=self.messages)
        
        for chunk in stream:
            if "choices" in chunk and chunk["choices"]:
                content = chunk["choices"][0].get("delta", {}).get("content", "")
                if content:
                    # Update usage if available
                    if "usage" in chunk:
                        self._usage.prompt_tokens += chunk["usage"].get("prompt_tokens", 0)
                        self._usage.completion_tokens += chunk["usage"].get("completion_tokens", 0)
                        self._usage.total_tokens += chunk["usage"].get("total_tokens", 0)
                    
                    # Add content to parts manager
                    self._parts_manager.append_part(content)
                    
                    # Yield the stream event
                    yield ModelResponseStreamEvent(
                        type="content",
                        content=content
                    )

    def timestamp(self) -> datetime:
        """Get timestamp of the response."""
        return self._timestamp
