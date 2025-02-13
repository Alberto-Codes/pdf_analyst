from config.state import GraphState
from graph.gemini_graph import gemini_graph
from models.sec_filing import SecFiling
from nodes.configure_api import ConfigureAPI
from utils.schema_utils import get_response_schema_from_model

from typing import Dict, Any
def create_vertex_schema(pydantic_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Pydantic schema to Vertex AI compatible format."""
    def resolve_ref(ref: str, definitions: Dict) -> Dict:
        """Resolve $ref references in the schema."""
        if not ref.startswith('#/$defs/'):
            return {}
        model_name = ref.split('/')[-1]
        return definitions.get(model_name, {})

    def convert_properties(schema_properties: Dict, definitions: Dict) -> Dict:
        properties = {}
        for prop_name, prop_info in schema_properties.items():
            if "$ref" in prop_info:
                # Handle references to other models
                ref_model = resolve_ref(prop_info["$ref"], definitions)
                nested_props = convert_properties(
                    ref_model.get("properties", {}), 
                    definitions
                )
                if nested_props:
                    properties[prop_name] = {
                        "type": "object",
                        "properties": nested_props
                    }
            elif prop_info.get("type") == "object":
                # Handle nested objects
                nested_props = convert_properties(
                    prop_info.get("properties", {}), 
                    definitions
                )
                if nested_props:
                    properties[prop_name] = {
                        "type": "object",
                        "properties": nested_props
                    }
            elif prop_info.get("type") == "array":
                # Handle arrays
                items = prop_info.get("items", {})
                if "$ref" in items:
                    ref_model = resolve_ref(items["$ref"], definitions)
                    item_props = convert_properties(
                        ref_model.get("properties", {}), 
                        definitions
                    )
                    properties[prop_name] = {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": item_props
                        }
                    }
                else:
                    properties[prop_name] = {
                        "type": "array",
                        "items": {"type": items.get("type", "string").lower()}
                    }
            else:
                # Handle primitive types
                properties[prop_name] = {
                    "type": prop_info.get("type", "string").lower()
                }
        return properties

    definitions = pydantic_schema.get("$defs", {})
    schema = {
        "type": "object",
        "properties": convert_properties(
            pydantic_schema.get("properties", {}),
            definitions
        ),
        "required": pydantic_schema.get("required", [])
    }
    
    return schema

# Debug the schema
import json
pydantic_schema = SecFiling.model_json_schema()
vertex_schema = create_vertex_schema(pydantic_schema)
print("Vertex Schema:", json.dumps(vertex_schema, indent=2))
if __name__ == "__main__":
    """Executes the Gemini graph synchronously with an initialized state.

    This script runs the `gemini_graph` synchronously, starting from the
    `ConfigureAPI` node. It initializes a `GraphState` object with necessary
    parameters, including the document URL, MIME type, prompt, and response schema.
    The execution of the graph results in content generation, with the generated
    response and execution history returned.

    The script extracts information from a PDF document using the Gemini API,
    focusing on retrieving employee count details.

    Process:
        1. Initializes `GraphState` with API parameters.
        2. Runs the Gemini graph synchronously, beginning with `ConfigureAPI`.
        3. Retrieves the generated response and execution history.

    Attributes:
        result (str): The generated content response from the Gemini API.
        history (list): The execution history of the graph, tracking all
            executed nodes in sequence.
    """
    # Initialize the state for the content generation process
    state = GraphState(
        document_url="https://www.wellsfargo.com/assets/pdf/about/investor-relations/sec-filings/2023/10k.pdf",
        document_mime_type="application/pdf",
        prompt="""
You are a document extraction expert specialized in SEC filings. Analyze the provided document and extract information according to the schema tags provided. Focus on company details, officer information, and filing metadata.

For each extraction, provide:
- Complete context with citations for found information
- Standard null values when information isn't found (page: 0, context: "Information not found in document", confidence: 0.0)
- Strong confidence (>0.9) for exact matches of EINs, names, dates, and titles
- Location data (bbox) when available

Pay special attention to document sections typically containing:
- Company identifiers and legal names
- Officer signatures and titles
- Filing dates and attestations""",
        response_mime_type="application/json",
        response_schema=vertex_schema,
        export_file_name=SecFiling.__name__.lower(),
    )

    # Run the Gemini graph synchronously with the initialized state
    result, history = gemini_graph.run_sync(
        ConfigureAPI(), state=state  # Pass the state object with necessary parameters
    )
