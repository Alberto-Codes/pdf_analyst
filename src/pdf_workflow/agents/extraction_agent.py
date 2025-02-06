import json

from agno.agent import Agent, AgentKnowledge
from agno.models.ollama import Ollama


def generate_template(model_cls) -> str:
    """Generate a JSON template string from a Pydantic V2 model class."""

    def recurse(cls):
        if hasattr(cls, "model_fields"):
            result = {}
            for field_name, field in cls.model_fields.items():
                field_type = field.annotation
                if hasattr(field_type, "__origin__") and field_type.__origin__ is list:
                    result[field_name] = (
                        [recurse(field_type.__args__[0])]
                        if hasattr(field_type.__args__[0], "model_fields")
                        else [field_name]
                    )
                elif hasattr(field_type, "model_fields"):
                    result[field_name] = recurse(field_type)
                else:
                    result[field_name] = field_name
            return result
        return str(cls)

    return json.dumps(recurse(model_cls), indent=2)


def process_document(knowledge: AgentKnowledge, model_cls):
    """Use an AI agent to extract structured information from the text."""
    template = generate_template(model_cls)

    agent = Agent(
        model=Ollama(id="llama3.2-3b-instruct-fp16-32k"),
        description=f"You are a precise document analyzer skilled at extracting and structuring information from your knowledge_base",
        instructions=["Search your knowledge base", "Answer the question"],
        # structured_outputs=True,
        markdown=True,
        knowledge=knowledge,
        read_chat_history=True,
        show_tool_calls=True,
        search_knowledge=True,
        add_references=False,
        debug_mode=True,
        additional_context="a director can have a title of chief executive officer, chief financial officer, chief operating officer, president, or vice president as some examples of directors.",
        response_model=model_cls,
    )

    extraction_prompt = f"""Extract the all of the directors listed information from your knowledge_base into the following JSON template exactly:
{template}
Do not output any extra text.
"""

    agent.print_response("can I get a list of all the direcotrs", stream=True)
