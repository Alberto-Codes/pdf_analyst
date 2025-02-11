import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from config.state import GraphState
from pydantic_graph import BaseNode, End, GraphRunContext


@dataclass
class ExportToCSV(BaseNode[GraphState]):
    """Node to export JSON response data to a CSV file.

    This node reads a JSON-formatted response from the graph state, converts
    it into CSV format, and saves it to a designated export directory. The
    file is named using a timestamp to ensure uniqueness.
    """

    async def run(self, ctx: GraphRunContext[GraphState]) -> End[Path]:
        """Executes the CSV export process.

        This method retrieves the JSON response from the graph state, parses
        it, and writes it to a CSV file in an export directory. If the JSON
        response is invalid, an empty path is returned.

        Args:
            ctx (GraphRunContext[GraphState]): The execution context containing
                the state with export directory, filename, and response data.

        Returns:
            End[Path]: The path to the created CSV file, or an empty path in
            case of a JSON parsing error.
        """
        export_dir = ctx.state.export_dir / f"{ctx.state.export_file_name}_exports"
        export_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = export_dir / f"{ctx.state.export_file_name}_{timestamp}.csv"

        if ctx.state.response_text:
            try:
                # Parse the JSON response
                response_data = json.loads(
                    ctx.state.response_text.replace("Gemini Response: ", "")
                )

                with open(filename, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=response_data.keys())
                    writer.writeheader()
                    writer.writerow(response_data)

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                # Handle the error appropriately, e.g., log it or return an error End
                return End(Path(""))  # Return an empty path to indicate failure

        return End(filename)
