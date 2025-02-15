from typing import Any, Dict
from dataclasses import dataclass
from google.genai import types
from pydantic_graph import BaseNode, GraphRunContext, End
from nodes.execute_api import ExecuteAPI
from utils.model_armor.sanitize_prompt import ModelArmorSanitizer
from config.state import GraphState

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

    def __init__(self, config: Dict[str, Any]):
        self.sanitizer = ModelArmorSanitizer(
            project_id=config.get("project_id"),
            location=config.get("location", "europe-west4"),
            template_id=config.get("template_id")
        )

    async def run(self, ctx: GraphRunContext[GraphState]) -> ExecuteAPI | End:
        try:
            # Get prompt text from last part
            prompt_part = ctx.state.contents[-1]
            if not isinstance(prompt_part, types.Part) or not prompt_part.text:
                raise ValueError("Invalid or missing prompt text")

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
                    return ExecuteAPI()
                case unknown:
                    return self._error(
                        "sanitization_error",
                        "Unexpected Model Armor response",
                        f"Unknown filterMatchState: {unknown}"
                    )

        except Exception as e:
            return self._error("sanitization_error", "Error during content sanitization", str(e))

    def _error(self, type_: str, message: str, details: Any, filter_results: dict = None) -> End:
        """Helper to create error state and return End node."""
        error = {"type": type_, "message": message, "details": details}
        if filter_results:
            error["filter_results"] = filter_results
        self.ctx.state.error = error
        return End()