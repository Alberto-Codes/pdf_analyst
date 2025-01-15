import os
import requests
import zipfile
from pathlib import Path

# URL to download the dataset
ZIP_URL = "https://zenodo.org/records/4595826/files/CUAD_v1.zip?download=1"
ZIP_DIR = Path("data/pdfs")  # Destination directory
ZIP_PATH = ZIP_DIR / "CUAD_v1.zip"  # Path to store the zip file
EXTRACT_DIR = ZIP_DIR  # Extract files into 'data/pdfs/'


def download_and_unzip():
    """
    Downloads the CUAD_v1.zip dataset from the specified URL and extracts it 
    into the 'data/pdfs/' directory.

    The function ensures that the target directory exists, downloads the 
    dataset, saves it as a ZIP file, and then extracts its contents.

    Raises:
        requests.exceptions.RequestException: If an error occurs during 
            the download process.
        zipfile.BadZipFile: If the downloaded file is not a valid ZIP archive.
        OSError: If an error occurs while writing the file or extracting it.
    """
    ZIP_DIR.mkdir(parents=True, exist_ok=True)  # Ensure 'data/pdfs/' exists

    # Download the ZIP file
    print(f"Downloading CUAD_v1.zip to {ZIP_PATH}...")
    try:
        response = requests.get(ZIP_URL, stream=True)
        response.raise_for_status()  # Raise an error for bad responses (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return

    # Save the ZIP file
    try:
        with open(ZIP_PATH, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Downloaded {ZIP_PATH}")
    except OSError as e:
        print(f"Error writing file: {e}")
        return

    # Extract ZIP file
    print(f"Extracting to {EXTRACT_DIR}...")
    try:
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
        print("Extraction complete.")
    except zipfile.BadZipFile as e:
        print(f"Error extracting ZIP file: {e}")
        return
    except OSError as e:
        print(f"Error during extraction: {e}")
        return


# Run the function
if __name__ == "__main__":
    download_and_unzip()
