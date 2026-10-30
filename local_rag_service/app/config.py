import ipaddress
import re
from pathlib import Path
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
EMBED_MODELS = ("BAAI/bge-m3", "intfloat/multilingual-e5-large")
LLM_MODELS = (
    "qwen3.5:9b",
    "qwen3.8:27b",
    "gemma4:12b",
    "gemma4:31b",
    "ministral-3:8b",
    "ministral-3:14b",
)


def sanitize(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", value)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    DEFAULT_LLM_MODEL: str = "qwen3.5:9b"
    JUDGE_MODEL: str = "gemma4:31b"
    DEFAULT_EMBED_MODEL: str = "BAAI/bge-m3"
    DEFAULT_RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    CHROMA_PERSIST_DIR: Path = BASE_DIR / "data/chroma_db"
    INDEX_DIR: Path = BASE_DIR / "data/indexes"
    RAW_DATA_DIR: Path = BASE_DIR / "data/raw"
    PROCESSED_MD_DIR: Path = BASE_DIR / "data/processed_md"
    METADATA_CSV_PATH: Path = BASE_DIR / "data/metadata.csv"
    LOG_DIR: Path = BASE_DIR / "logs"
    MODEL_DIR: Path = BASE_DIR / "models"
    DEFAULT_TEMPERATURE: float = 0.1
    DEFAULT_TOP_K: int = 3
    HOST: str = "0.0.0.0"
    PORT: int = Field(default=8000, ge=1, le=65535)
    DEVICE: str = "cpu"
    EMBED_BATCH_SIZE: int = Field(default=8, ge=1)
    OLLAMA_TIMEOUT: float = Field(default=120.0, gt=0)
    OLLAMA_NUM_CTX: int = Field(default=32768, ge=2048)
    OLLAMA_NUM_PREDICT: int = Field(default=2048, ge=128)
    OLLAMA_KEEP_ALIVE: str = "5m"
    SEED: int = 42
    OCR_ENABLED: bool = False
    OCR_LANGUAGES: list[str] = ["rus", "eng"]
    REQUIRE_OLLAMA_ON_STARTUP: bool = False

    @model_validator(mode="after")
    def context_budget(self):
        if self.OLLAMA_NUM_PREDICT + 256 >= self.OLLAMA_NUM_CTX:
            raise ValueError("OLLAMA_NUM_CTX must exceed OLLAMA_NUM_PREDICT + 256")
        return self

    @field_validator("OLLAMA_BASE_URL")
    @classmethod
    def local_ollama(cls, value: str) -> str:
        parsed = urlparse(value)
        host = parsed.hostname or ""
        if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password:
            raise ValueError("Ollama URL must be a local HTTP(S) address without credentials")
        if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            raise ValueError("Ollama URL must not contain a path, query or fragment")
        if host != "localhost":
            address = ipaddress.ip_address(host)
            if not (address.is_private or address.is_loopback) or address.is_unspecified:
                raise ValueError("Ollama must use localhost or a private IP address")
        return value.rstrip("/")

    @field_validator(
        "CHROMA_PERSIST_DIR",
        "INDEX_DIR",
        "RAW_DATA_DIR",
        "PROCESSED_MD_DIR",
        "METADATA_CSV_PATH",
        "LOG_DIR",
        "MODEL_DIR",
        mode="after",
    )
    @classmethod
    def absolute_path(cls, value: Path) -> Path:
        return (BASE_DIR / value).resolve() if not value.is_absolute() else value.resolve()

    def model_path(self, model: str) -> Path:
        path = self.MODEL_DIR / sanitize(model)
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"Local model missing: {path}. Run prepare_models.py online.")
        return path

    def create_directories(self) -> None:
        for path in (
            self.CHROMA_PERSIST_DIR,
            self.INDEX_DIR,
            self.RAW_DATA_DIR,
            self.PROCESSED_MD_DIR,
            self.LOG_DIR,
            self.METADATA_CSV_PATH.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)
