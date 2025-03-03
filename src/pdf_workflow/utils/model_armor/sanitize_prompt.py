import os
from typing import Dict, Any

MOCK_RESPONSES = {
    "success": {
        "sanitizationResult": {
            "invocationResult": "SUCCESS",
            "filterMatchState": "NO_MATCH_FOUND"
        }
    },
    "blocked_rai": {
        "sanitizationResult": {
            "invocationResult": "SUCCESS",
            "filterMatchState": "MATCH_FOUND",
            "filterResults": {
                "rai": {"score": 0.9}
            }
        }
    },
    "blocked_multiple": {
        "sanitizationResult": {
            "invocationResult": "SUCCESS",
            "filterMatchState": "MATCH_FOUND",
            "filterResults": {
                "rai": {"score": 0.9},
                "sdp": {"score": 0.8},
                "pi_and_jailbreak": {"score": 0.95}
            }
        }
    },
    "api_error": {
        "sanitizationResult": {
            "invocationResult": "ERROR",
            "errorMessage": "Internal server error"
        }
    }
}

class ModelArmorSanitizer:
    def __init__(self, project_id: str, location: str = "europe-west4", template_id: str = None):
        self.mock_response = os.getenv("MODEL_ARMOR_MOCK", "success")

    def sanitize_prompt(self, content: str) -> Dict[str, Any]:
        """Simulate Model Armor API response for development
        
        Set MODEL_ARMOR_MOCK environment variable to one of:
        - success: Content passes all checks
        - blocked_rai: Content blocked for responsible AI concerns
        - blocked_multiple: Content blocked for multiple reasons
        - api_error: API returns an error
        """
        # For development, you could also trigger different responses based on content
        if "trigger_block" in content.lower():
            return MOCK_RESPONSES["blocked_rai"]
        if "trigger_error" in content.lower():
            return MOCK_RESPONSES["api_error"]
        
        return MOCK_RESPONSES.get(self.mock_response, MOCK_RESPONSES["success"]) 
