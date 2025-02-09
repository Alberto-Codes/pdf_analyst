import base64


def encode_file(file_path: str, encoding: str = "utf-8") -> str:
    """
    Read and encode a file to base64.

    Args:
        file_path (str): Path to the file
        encoding (str): Encoding to use for the result

    Returns:
        str: Base64 encoded content

    Raises:
        FileNotFoundError: If file is not found
        IOError: If there's an error reading the file
    """
    try:
        with open(file_path, "rb") as file:
            data = file.read()
            return base64.b64encode(data).decode(encoding)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at path: {file_path}")
    except IOError as e:
        raise IOError(f"Error reading file: {str(e)}")
