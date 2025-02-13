from typing import Any, Dict


def create_vertex_schema(pydantic_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Pydantic schema to Vertex AI compatible format."""

    def resolve_ref(ref: str, definitions: Dict) -> Dict:
        """Resolve $ref references in the schema."""
        if not ref.startswith("#/$defs/"):
            return {}
        model_name = ref.split("/")[-1]
        return definitions.get(model_name, {})

    def convert_properties(schema_properties: Dict, definitions: Dict) -> Dict:
        properties = {}
        for prop_name, prop_info in schema_properties.items():
            property_schema = {}

            if "$ref" in prop_info:
                ref_model = resolve_ref(prop_info["$ref"], definitions)
                nested_props = convert_properties(
                    ref_model.get("properties", {}), definitions
                )
                if nested_props:
                    property_schema.update(
                        {"type": "object", "properties": nested_props}
                    )
            elif prop_info.get("type") == "object":
                nested_props = convert_properties(
                    prop_info.get("properties", {}), definitions
                )
                if nested_props:
                    property_schema.update(
                        {"type": "object", "properties": nested_props}
                    )
                    if "required" in prop_info:
                        property_schema["required"] = prop_info["required"]
                    if "minProperties" in prop_info:
                        property_schema["minProperties"] = str(
                            prop_info["minProperties"]
                        )
                    if "maxProperties" in prop_info:
                        property_schema["maxProperties"] = str(
                            prop_info["maxProperties"]
                        )
                    if "propertyOrdering" in prop_info:
                        property_schema["propertyOrdering"] = prop_info[
                            "propertyOrdering"
                        ]
            elif prop_info.get("type") == "array":
                items = prop_info.get("items", {})
                if "$ref" in items:
                    ref_model = resolve_ref(items["$ref"], definitions)
                    item_props = convert_properties(
                        ref_model.get("properties", {}), definitions
                    )
                    property_schema.update(
                        {
                            "type": "array",
                            "items": {"type": "object", "properties": item_props},
                        }
                    )
                else:
                    property_schema.update(
                        {
                            "type": "array",
                            "items": {"type": items.get("type", "string").lower()},
                        }
                    )
                if "minItems" in prop_info:
                    property_schema["minItems"] = str(prop_info["minItems"])
                if "maxItems" in prop_info:
                    property_schema["maxItems"] = str(prop_info["maxItems"])
            else:
                type_val = prop_info.get("type", "string").lower()
                property_schema["type"] = type_val

                # Handle formats for different types
                if type_val == "number":
                    property_schema["format"] = prop_info.get("format", "double")
                elif type_val == "integer":
                    property_schema["format"] = prop_info.get("format", "int64")
                elif type_val == "string" and "enum" in prop_info:
                    property_schema["format"] = "enum"
                elif type_val == "string" and "format" in prop_info:
                    property_schema["format"] = prop_info["format"]

            # Add common schema properties
            if "title" in prop_info:
                property_schema["title"] = prop_info["title"]
            if "description" in prop_info:
                property_schema["description"] = prop_info["description"]
            if "nullable" in prop_info:
                property_schema["nullable"] = prop_info["nullable"]
            if "default" in prop_info:
                property_schema["default"] = prop_info["default"]
            if "enum" in prop_info:
                property_schema["enum"] = [str(v) for v in prop_info["enum"]]
            # Handle both example and examples
            if "example" in prop_info:
                property_schema["example"] = prop_info["example"]
            elif "examples" in prop_info:
                # Take the first example if multiple are provided
                examples = prop_info["examples"]
                if examples and len(examples) > 0:
                    property_schema["example"] = examples[0]

            # Type-specific validations
            if prop_info.get("type") == "string":
                if "minLength" in prop_info:
                    property_schema["minLength"] = str(prop_info["minLength"])
                if "maxLength" in prop_info:
                    property_schema["maxLength"] = str(prop_info["maxLength"])
                if "pattern" in prop_info:
                    property_schema["pattern"] = prop_info["pattern"]
            elif prop_info.get("type") in ["number", "integer"]:
                if "minimum" in prop_info:
                    property_schema["minimum"] = float(prop_info["minimum"])
                if "maximum" in prop_info:
                    property_schema["maximum"] = float(prop_info["maximum"])

            properties[prop_name] = property_schema

        return properties

    # Define definitions before use
    definitions = pydantic_schema.get("$defs", {})

    # Create base schema
    schema = {
        "type": "object",
        "properties": convert_properties(
            pydantic_schema.get("properties", {}), definitions
        ),
    }

    # Add optional top-level properties
    for field in ["title", "description", "example", "required", "anyOf"]:
        if field in pydantic_schema:
            schema[field] = pydantic_schema[field]

    return schema
