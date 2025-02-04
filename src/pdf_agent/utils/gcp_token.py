import os

import google.auth.transport.requests
import google.oauth2.id_token


def fetch_gcp_id_token():
    """Fetches an ID token for an authorized GET request.

    This function reads the GCP endpoint from the `GCP_OLLAMA_ENDPOINT`
    environment variable and generates an ID token using Google authentication
    libraries. The token is intended for use in requests to secure endpoints,
    such as those hosted on Cloud Run.

    Raises:
        ValueError: If the `GCP_OLLAMA_ENDPOINT` environment variable is not set.

    Returns:
        str: A Google-signed ID token for the specified endpoint.
    """
    # Read the endpoint from the environment variable
    endpoint = os.getenv("GCP_OLLAMA_ENDPOINT")
    if not endpoint:
        raise ValueError("GCP_OLLAMA_ENDPOINT environment variable is not set")

    # Cloud Run uses your service's hostname as the `audience` value
    audience = endpoint

    auth_req = google.auth.transport.requests.Request()
    id_token = google.oauth2.id_token.fetch_id_token(auth_req, audience)

    return id_token
