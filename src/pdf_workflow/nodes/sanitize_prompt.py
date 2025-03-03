from typing import Any, Dict
from dataclasses import dataclass
import logging

from google.genai import types
from pydantic_graph import BaseNode, GraphRunContext, End

from pdf_workflow.config.state import GraphState
from pdf_workflow.nodes.execute_api import ExecuteAPI
from pdf_workflow.utils.model_armor.sanitize_prompt import ModelArmorSanitizer


FILTER_ERROR_MESSAGES = {
    "rai": "Responsible AI concerns detected",
    "sdp": "Sensitive data detected",
    "pi_and_jailbreak": "Prompt injection or jailbreak attempt detected",
    "malicious_uris": "Malicious URLs detected",
    "csam": "Content safety violation detected"
}


@dataclass
class SanitizePrompt(BaseNode[GraphState]):
    """Sanitizes prompt content using Google Cloud Model Armor."""

    config: Dict[str, Any]

    def __post_init__(self):
        """Initialize the sanitizer after class initialization."""
        self.sanitizer = ModelArmorSanitizer(
            project_id=self.config.get("project_id"),
            location=self.config.get("location", "europe-west4"),
            template_id=self.config.get("template_id")
        )
        self.ctx = None

    async def run(self, ctx: GraphRunContext[GraphState]) -> ExecuteAPI | End:
        """Sanitizes the prompt content using Model Armor.
        
        Args:
            ctx (GraphRunContext[GraphState]): The execution context containing
                the shared state with prompt content.
                
        Returns:
            ExecuteAPI | End: Returns either the next node in the workflow
                or terminates the workflow if content is blocked.
        """
        # Store context for error handling
        self.ctx = ctx
        
        try:
            # Get prompt text from last part
            prompt_part = ctx.state.contents[-1]
            if not isinstance(prompt_part, types.Part) or not prompt_part.text:
                raise ValueError("Invalid or missing prompt text")

            logging.info("Sanitizing prompt content")
            result = self.sanitizer.sanitize_prompt(prompt_part.text)
            
            # Check API success
            if result.get("sanitizationResult", {}).get("invocationResult") != "SUCCESS":
                return self._error("sanitization_error", "Model Armor API invocation failed", result)

            match result["sanitizationResult"]["filterMatchState"]:
                case "MATCH_FOUND":
                    filter_results = result["sanitizationResult"]["filterResults"]
                    error_details = [
                        msg for filter_name, msg in FILTER_ERROR_MESSAGES.items()
                        if filter_name in filter_results
                    ]
                    return self._error(
                        "content_blocked",
                        "Content was blocked by Model Armor",
                        " | ".join(error_details),
                        filter_results
                    )
                case "NO_MATCH_FOUND":
                    logging.info("Content passed sanitization checks")
                    return ExecuteAPI()
                case unknown:
                    return self._error(
                        "sanitization_error",
                        "Unexpected Model Armor response",
                        f"Unknown filterMatchState: {unknown}"
                    )

        except Exception as e:
            logging.error(f"Error during content sanitization: {str(e)}")
            return self._error("sanitization_error", "Error during content sanitization", str(e))

    def _error(self, type_: str, message: str, details: Any, filter_results: dict = None) -> End:
        """Helper to create error state and return End node.
        
        Args:
            type_: The type of error
            message: A descriptive message about the error
            details: Additional details about the error
            filter_results: Optional filter results from Model Armor
            
        Returns:
            End: End node to terminate the workflow
        """
        if self.ctx is None:
            logging.error(f"Error ({type_}): {message} - {details}")
            return End()
            
        error = {"type": type_, "message": message, "details": details}
        if filter_results:
            error["filter_results"] = filter_results
        self.ctx.state.error = error
        return End()
