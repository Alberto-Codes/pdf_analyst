from pydantic import BaseModel


def get_response_schema_from_model(model_class: type[BaseModel]) -> dict:
    """Convert a Pydantic model to Gemini API response schema format.

    This function converts a Pydantic model class into a format that can
    be used for the response schema in Gemini API. It extracts the model's
    JSON schema and constructs the required schema for the API, including
    the properties and the required fields.

    Args:
        model_class (type[BaseModel]): The Pydantic model class to extract
            the schema from.

    Returns:
        dict: The Gemini API-compatible schema, including properties
            and required fields.
    """
    schema = model_class.model_json_schema()

    return {
        "type": "object",
        "properties": schema["properties"],
        "required": schema.get("required", []),
    }
