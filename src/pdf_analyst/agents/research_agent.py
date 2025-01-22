import os
from typing import Optional

import google.auth.transport.requests
import google.oauth2.id_token
from phi.agent import Agent
from phi.knowledge.agent import AgentKnowledge
from phi.model.ollama import Ollama
from phi.storage.agent.sqlite import SqlAgentStorage
# from phi.tools.duckduckgo import DuckDuckGo
from phi.vectordb.chroma import ChromaDb
# from phi.document.chunking.agentic import AgenticChunking
from phi.document.chunking.fixed import FixedSizeChunking
from phi.embedder.ollama import OllamaEmbedder

# from ..models.embedder import EmbedderModel

# from agents.settings import agent_settings

research_agent_storage = SqlAgentStorage(
    table_name="research_agent_sessions", db_file="data/research_agent.sqlite"
)

research_agent_knowledge = AgentKnowledge(
    num_documents=10,
    vector_db=ChromaDb(
        collection="knowledge_base",
        embedder=OllamaEmbedder(
            # dimensions=2048, 
            model='nomic-embed-text',
            host=os.getenv("OLLAMA_HOST", "localhost"),
            port=8080,
            client_kwargs={
                "headers": {
                    "X-Serverless-Authorization": f"Bearer {google.oauth2.id_token.fetch_id_token(
                        google.auth.transport.requests.Request(), os.getenv('GCP_OLLAMA_ENDPOINT'))}"
                }
            }),
        path="./data/research_agent_knowledge.db",
        persistent_client=True,
        # monitoring=True,
    ),
    # chunking_strategy=FixedSizeChunking(chunk_size=500, overlap=100),
)


def get_research_agent(
    model_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = False,
) -> Agent:
    """Creates and returns an instance of the Research Agent.

    The Research Agent is a highly advanced AI with access to an extensive
    knowledge base. It uses the provided model for inference and offers tools
    and instructions for interacting with users effectively.

    Args:
        model_id (Optional[str]): The identifier for the model to use. Defaults
            to `None`.
        user_id (Optional[str]): The identifier for the user. Defaults to
            `None`.
        session_id (Optional[str]): The identifier for the agent session.
            Defaults to `None`.
        debug_mode (bool): Whether to enable debug mode for the agent. Defaults
            to `False`.

    Returns:
        Agent: An instance of the Research Agent configured with the specified
        parameters.
    """
    return Agent(
        name="Research Agent",
        agent_id="research-agent",
        session_id=session_id,
        user_id=user_id,
        # The model to use for the agent
        model=Ollama(
            id=model_id,
            host=os.getenv("OLLAMA_HOST", "localhost"),
            port=8080,
            client_params={
                "headers": {
                    "X-Serverless-Authorization": f"Bearer {google.oauth2.id_token.fetch_id_token(
                        google.auth.transport.requests.Request(), os.getenv('GCP_OLLAMA_ENDPOINT'))}"
                }
            },
            options={
                "temperature": 0.5, # The temperature for sampling
                # "max_tokens": 100, # The maximum number of tokens to generate
            #     # "top_p": 0.9, # The nucleus sampling parameter
            #     # "top_k": 50, # The top-k sampling parameter
            #     # "presence_penalty": 0.0, # The presence penalty
            #     # "frequency_penalty": 0.0, # The frequency penalty
            #     # "best_of": 1, # The number of completions to generate and return
            },
        ),
        # Tools available to the agent
        # tools=[DuckDuckGo()],
        # A description of the agent that guides its overall behavior
        description="You are a highly advanced AI agent with access to an extensive knowledge base",
        # A list of instructions to follow, each as a separate item in the list
        instructions=[
            "Always search your knowledge base.\n"
            "  - Search your knowledge base for information.\n"
            "  - Provide answers based on your existing knowledge whenever possible.",
            "  - Make multiple searches of the knowledge base using different queries.",
            "Provide concise and relevant answers.\n"
            "  - Keep your responses brief and to the point.\n"
            "  - Focus on delivering the most pertinent information without unnecessary detail.",
            "Ask clarifying questions.\n"
            "  - If a user's request is unclear or incomplete, ask specific questions to obtain the necessary details.\n"
            "  - Ensure you fully understand the inquiry before formulating a response.",
            "Verify the information you provide for accuracy.",
            "Cite reliable sources from the knowledge base.",
            "Check if your response will answer the user's question.",
        ],
        task="Answer user queries searching your knowledge base. Perform as many knoweldge base searches as you need",
        # Format responses as markdown
        markdown=True,
        # Show tool calls in the response
        show_tool_calls=True,
        # Add the current date and time to the instructions
        add_datetime_to_instructions=True,
        # Store agent sessions in the database
        storage=research_agent_storage,
        # Enable read the chat history from the database
        read_chat_history=True,
        # Store knowledge in a vector database
        knowledge=research_agent_knowledge,
        # Enable searching the knowledge base
        search_knowledge=True,
        # Enable monitoring on phidata.app
        monitoring=True,
        # Show debug logs
        debug_mode=debug_mode,
        # show_intermediate_results=True,
        # stream=True,
        # stream_intermediate_results=True,
        # add_references=True,
        read_tool_call_history=True,
        structured_outputs=True,
        # reasoning=True,
        # reasoning_model=Ollama(
        #     id='deepseek-r1:7b_32k',
        #     # id=model_id,
        #     host=os.getenv("OLLAMA_HOST", "localhost"),
        #     port=8080,
        #     client_params={
        #         "headers": {
        #             "X-Serverless-Authorization": f"Bearer {google.oauth2.id_token.fetch_id_token(
        #                 google.auth.transport.requests.Request(), os.getenv('GCP_OLLAMA_ENDPOINT'))}"
        #         }
        #     },
        #     # options={"max_tokens": 100},
        #     add_datetime_to_instructions=True,
        #     search_knowledge=True,
        #     show_tool_calls=True,
        #     structured_outputs=True,
        #     monitoring=True,
        # ),

    )
