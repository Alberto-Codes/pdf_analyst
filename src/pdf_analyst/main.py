from phi.agent import Agent, RunResponse
from phi.model.ollama import Ollama
from phi.tools.duckduckgo import DuckDuckGo

web_agent = Agent(
    name="Web Agent",
    model=Ollama(id="llama3.2"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)
web_agent.print_response("Tell me about Warren Buffet?", stream=True)

# Get the response in a variable
run: RunResponse = web_agent.run("Tell me about Warren Buffet?")
print(run.content)
