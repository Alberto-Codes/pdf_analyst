import requests
from google.auth import default
from google.auth.transport.requests import Request

class ModelArmorSanitizer:
    def __init__(self, project_id: str, location: str = "europe-west4", template_id: str = None):
        template_id = template_id or "default-template"
        self.url = f"https://modelarmor.{location}.rep.googleapis.com/v1/projects/{project_id}/locations/{location}/templates/{template_id}:sanitizeUserPrompt"
        self.credentials, _ = default()

    def sanitize_prompt(self, content: str) -> dict:
        """Sanitize user prompt using Google Cloud Model Armor"""
        self.credentials.refresh(Request())
        response = requests.post(
            self.url,
            headers={
                "Authorization": f"Bearer {self.credentials.token}",
                "Content-Type": "application/json"
            },
            json={"user_prompt_data": {"text": content}}
        )
        response.raise_for_status()
        return response.json() 