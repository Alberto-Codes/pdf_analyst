from phi.knowledge.pdf import PDFUrlKnowledgeBase
from rich.prompt import Prompt
from pdf_analyst.utilities.config_loader import load_config
from .base_agent import BaseAgent

class PDFAgent(BaseAgent):
    def __init__(self, knowledge_base: PDFUrlKnowledgeBase):
        super().__init__()
        self.knowledge = knowledge_base
        self.configure_agent()
    
    def configure_agent(self):
        config = load_config()
        agent_config = config.get("agent", {})
        
        # Configure agent settings from config
        self.description = agent_config.get("description")
        self.task = agent_config.get("task")
        self.guidelines = agent_config.get("guidelines", [])
        self.instructions = agent_config.get("instructions", [])
        
        # Configure reasoning settings
        reasoning_config = agent_config.get("reasoning", {})
        self.reasoning = reasoning_config.get("enabled", False)
        self.reasoning_min_steps = reasoning_config.get("min_steps", 2)
        self.reasoning_max_steps = reasoning_config.get("max_steps", 6)
        
        # Configure other settings
        settings = agent_config.get("settings", {})
        self.use_tools = settings.get("use_tools", True)
        self.show_tool_calls = settings.get("show_tool_calls", True)
        self.markdown = settings.get("markdown", True)
        self.debug_mode = settings.get("debug_mode", True)
        self.stream = settings.get("stream", False)
        self.stream_intermediate_steps = settings.get("stream_intermediate_steps", False)