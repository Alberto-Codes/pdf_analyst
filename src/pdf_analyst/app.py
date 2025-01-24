import asyncio
import csv
import os

import httpx
from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from utilities.gcp_token import fetch_gcp_id_token


def print_year_and_url(csv_file_path: str):
    """
    Reads a CSV file and prints the 'year' and 'url' fields for each row.

    Args:
        csv_file_path (str): The file path to the CSV file containing the data.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        KeyError: If the 'year' or 'url' fields are not in the CSV file.
    """
    try:
        with open(csv_file_path, mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                year = row.get("year")
                url = row.get("url")
                print(f"Year: {year}, URL: {url}")
    except FileNotFoundError:
        print(f"File not found: {csv_file_path}")
    except KeyError as e:
        print(f"Missing field in CSV: {e}")


async def use_ollama_model() -> str:
    """
    Asynchronously interacts with an OllamaModel to generate a response.

    The function fetches a GCP ID token, sets up the necessary headers and
    HTTP client, and interacts with the OllamaModel to obtain a response.

    Returns:
        str: The result from the OllamaModel.

    Raises:
        httpx.RequestError: If the HTTP request fails.
        Exception: For other runtime exceptions during model interaction.
    """
    try:
        token = fetch_gcp_id_token()
        headers = {"X-Serverless-Authorization": f"Bearer {token}"}
        async_client = httpx.AsyncClient(headers=headers)

        model = OllamaModel(
            model_name="deepseek-r1:8b_32k",
            api_key="OLLAMA",
            base_url=os.getenv("GCP_OLLAMA_ENDPOINT") + "/v1",
            http_client=async_client,
        )
        agent = Agent(model=model)
        result = await agent.run(
            'Where does "hello world" come from? Answer in a very short sentence.'
        )
        return result
    except Exception as e:
        print(f"Error occurred: {e}")
        raise


def main():
    """
    Main function to execute the workflow.

    This function reads and prints the year and URL from a CSV file and
    asynchronously fetches a response from the OllamaModel.
    """
    csv_file_path = "seeds/wfc_10k.csv"
    print_year_and_url(csv_file_path)

    try:
        result = asyncio.run(use_ollama_model())
        print(result)
    except Exception as e:
        print(f"An error occurred in main: {e}")


if __name__ == "__main__":
    main()
