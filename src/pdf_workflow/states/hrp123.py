from pathlib import Path
from typing import Dict, List, Optional

from google import genai
from google.genai import types
from models.sec_filing import SecFiling
from pydantic import BaseModel, ConfigDict, HttpUrl


class Hrp123GraphState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: str = "gemini-2.0-flash-001"
    temperature: float = 0.4
    initial_document: SecFiling = None
    key_tag: str = None
    related_documents: List[SecFiling] = []
    config: types.GenerateContentConfig | None = None
    client: genai.Client | None = None
    export_dir: Path = Path("data")
    export_model_name: str | None = None
