from __future__ import annotations

from graphs.yra import gemini_graph
from models.sec_filing import SecFiling
from nodes.configure_api import ConfigureAPI
from states.hrp123 import Hrp123GraphState
from utils.schema_utils import get_response_schema_from_model


async def run_gemini_graph():
    state = Hrp123GraphState(
        initial_document=SecFiling(file_uri="https://www.wellsfargo.com/assets/pdf/about/investor-relations/sec-filings/2023/10k.pdf")
    )

    # Run the Gemini graph synchronously with the initialized state
    await gemini_graph.run(
        ConfigureAPI(), state=state  # Pass the state object with necessary parameters
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(run_gemini_graph())
