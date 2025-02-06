import urllib.parse


def is_valid_url(url: str) -> bool:
    """Check if the provided string is a valid URL and ends with .pdf."""
    try:
        result = urllib.parse.urlparse(url)
        return all([result.scheme, result.netloc]) and url.lower().endswith(".pdf")
    except:
        return False
