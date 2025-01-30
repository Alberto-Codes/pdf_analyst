# watsonx_integration.py

from pydantic_ai.models import (
    Model, AgentModel, override_allow_model_requests,
    check_allow_model_requests,
)
from pydantic_ai.types import (
    ModelMessage, ModelSettings, ModelResponse, Usage,
    ModelResponseStreamEvent, StreamedResponse,
)
from pydantic import BaseModel, Field

from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextGenParameters, TextChatParameters
from ibm_watsonx_ai import Credentials

import datetime
from typing import AsyncIterator, Tuple, Optional, Dict, Any, List
import asyncio


class WatsonXParameters(BaseModel):
    """Parameters for WatsonX model requests"""

    # Common generation parameters
    temperature: float = Field(0.7, ge=0, le=1)
    max_new_tokens: int = Field(100, ge=1)
    min_new_tokens: Optional[int] = Field(None, ge=1)
    top_p: float = Field(1, ge=0, le=1)
    top_k: Optional[int] = Field(None, ge=1)
    random_seed: Optional[int] = None
    repetition_penalty: Optional[float] = Field(None, ge=0)
    decoding_method: str = Field("greedy", regex="^(greedy|sample)$")

    # Additional advanced fields from IBM WatsonX docs
    frequency_penalty: Optional[float] = Field(None, ge=0)
    presence_penalty: Optional[float] = Field(None, ge=0)
    n: Optional[int] = Field(None, ge=1, description="Number of responses")
    time_limit: Optional[int] = Field(None, ge=1)
    stop_sequences: Optional[List[str]] = None

    # Optionally returning logs / tokens, if used
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = Field(None, ge=1)

    # For Chat, we can define response_format or length_penalty, if needed
    # presence_penalty / frequency_penalty also exist in Chat


class WatsonXModel(Model):
    """
    Custom pydantic-ai Model integration for IBM WatsonX AI with extended features.
    """

    def __init__(
        self,
        api_key: str,
        service_url: str,
        project_id: str = None,
        model_id: str = None,
        parameters: Optional[WatsonXParameters] = None,
        allow_model_requests: bool = True,
    ):
        # Ensure external requests are allowed
        check_allow_model_requests()
        
        self.api_key = api_key
        self.service_url = service_url
        self.project_id = project_id
        self.model_id = model_id
        self.parameters = parameters or WatsonXParameters()

    async def agent_model(
        self,
        *,
        function_tools: list,
        allow_text_result: bool,
        result_tools: list,
    ) -> AgentModel:
        check_allow_model_requests()

        creds = Credentials(
            api_key=self.api_key,
            url=self.service_url,
        )
        
        # Convert function tools to WatsonX format
        watsonx_tools = self._convert_tools(function_tools + result_tools)
        
        model_inference = ModelInference(
            model_id=self.model_id,
            credentials=creds,
            project_id=self.project_id,
            params=self.parameters.dict(exclude_none=True)
        )

        return WatsonXAgentModel(
            model_inference=model_inference,
            tools=watsonx_tools,
            allow_text_result=allow_text_result
        )
    
    def _convert_tools(self, function_tools: list) -> List[Dict[str, Any]]:
        """Convert pydantic-ai function tools to WatsonX format."""
        watsonx_tools = []
        for tool in function_tools:
            watsonx_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            param.name: {
                                "type": self._get_param_type(param.type),
                                "description": param.description
                            }
                            for param in tool.parameters
                        },
                        "required": [param.name for param in tool.parameters if param.required]
                    }
                }
            }
            watsonx_tools.append(watsonx_tool)
        return watsonx_tools
    
    def _get_param_type(self, param_type: str) -> str:
        """Convert Python type hints to JSON Schema types."""
        type_mapping = {
            'str': 'string',
            'int': 'integer',
            'float': 'number',
            'bool': 'boolean',
            'list': 'array',
            'dict': 'object'
        }
        return type_mapping.get(param_type, 'string')


class WatsonXAgentModel(AgentModel):
    """Per-step model for the agent with extended WatsonX functionality."""
    
    def __init__(
        self,
        model_inference: ModelInference,
        tools: List[Dict[str, Any]],
        allow_text_result: bool
    ):
        self.model_inference = model_inference
        self.tools = tools
        self.allow_text_result = allow_text_result
        self._timestamp = datetime.datetime.utcnow()

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
    ) -> Tuple[ModelResponse, Usage]:
        watsonx_messages = self._convert_messages(messages)
        
        try:
            if len(watsonx_messages) > 1 or not self.allow_text_result:
                # Use chat endpoint with tools
                response = await self.model_inference.achat(
                    messages=watsonx_messages,
                    tools=self.tools if self.tools else None
                )
                model_response = self._process_chat_response(response)
            else:
                # Single prompt => use generate endpoint
                response = await self.model_inference.agenerate(
                    prompt=watsonx_messages[0]['content']
                )
                model_response = self._process_generate_response(response)
                
            # Extract usage
            usage = self._extract_usage(response)
            
            return model_response, usage
        except Exception as e:
            raise RuntimeError(f"WatsonX request failed: {str(e)}")

    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
    ) -> AsyncIterator[StreamedResponse]:
        watsonx_messages = self._convert_messages(messages)
        
        if len(watsonx_messages) > 1 or not self.allow_text_result:
            stream = self.model_inference.chat_stream(
                messages=watsonx_messages,
                tools=self.tools if self.tools else None
            )
        else:
            stream = self.model_inference.generate_text_stream(
                prompt=watsonx_messages[0]['content']
            )
            
        return WatsonXStreamedResponse(stream)

    def _convert_messages(self, messages: list[ModelMessage]) -> list[dict]:
        """Map pydantic-ai roles to WatsonX roles, including function calls."""
        role_mapping = {
            "system": "system",
            "user": "user",
            "assistant": "assistant",
            "tool": "function"  # If the model calls a 'tool' message
        }
        return [
            {
                "role": role_mapping.get(msg.role, msg.role),
                "content": msg.content,
                **({"name": msg.name} if hasattr(msg, 'name') else {})
            }
            for msg in messages
        ]

    def _process_chat_response(self, response: Dict[str, Any]) -> ModelResponse:
        message = response['choices'][0]['message']
        content = message.get('content', '')
        function_call = message.get('function_call')

        if function_call:
            return ModelResponse(
                content=content,
                role="assistant",
                function_call={
                    "name": function_call["name"],
                    "arguments": function_call["arguments"]
                },
                finish_reason="function_call"
            )
        
        return ModelResponse(
            content=content,
            role="assistant",
            finish_reason="stop"
        )

    def _process_generate_response(self, response: Dict[str, Any]) -> ModelResponse:
        return ModelResponse(
            content=response['results'][0]['generated_text'],
            role="assistant",
            finish_reason="stop"
        )

    def _extract_usage(self, response: Dict[str, Any]) -> Usage:
        usage_data = response.get('usage', {})
        return Usage(
            prompt_tokens=usage_data.get('prompt_tokens', 0),
            completion_tokens=usage_data.get('completion_tokens', 0),
            total_tokens=usage_data.get('total_tokens', 0)
        )

    @property
    def timestamp(self) -> datetime.datetime:
        return self._timestamp


class WatsonXStreamedResponse(StreamedResponse):
    """Enhanced streaming response handler for WatsonX."""
    
    def __init__(self, watsonx_stream_generator):
        self._stream_generator = watsonx_stream_generator
        self._collected_content = []
        self._function_calls = []
        self._usage = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        self._model_name = "ibm-watsonx-ai"
        self._timestamp = datetime.datetime.utcnow()

    async def __aiter__(self) -> AsyncIterator[ModelResponseStreamEvent]:
        async for chunk in self._stream_generator:
            delta = chunk['choices'][0]['delta']
            
            # Check if there's a partial function call
            if function_call := delta.get('function_call'):
                self._function_calls.append(function_call)
                yield ModelResponseStreamEvent(
                    content="",
                    role="assistant",
                    function_call=function_call,
                    finish_reason=None
                )
            
            # Handle partial content
            if content := delta.get('content', ''):
                self._collected_content.append(content)
                yield ModelResponseStreamEvent(
                    content=content,
                    role="assistant",
                    finish_reason=None
                )

    def get(self) -> ModelResponse:
        content = "".join(self._collected_content)
        if self._function_calls:
            function_call = self._function_calls[-1]
            return ModelResponse(
                content=content,
                role="assistant",
                function_call=function_call,
                finish_reason="function_call"
            )
        return ModelResponse(
            content=content,
            role="assistant",
            finish_reason="stop"
        )

    def usage(self) -> Usage:
        return self._usage

    def model_name(self) -> str:
        return self._model_name

    def timestamp(self) -> datetime.datetime:
        return self._timestamp
