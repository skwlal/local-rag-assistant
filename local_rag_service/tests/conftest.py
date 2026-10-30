import socket

import pytest

from app.config import Settings


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Unit/integration tests must not access the network")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)


@pytest.fixture
def settings(tmp_path):
    result = Settings(
        _env_file=None,
        RAW_DATA_DIR=tmp_path / "raw",
        PROCESSED_MD_DIR=tmp_path / "md",
        CHROMA_PERSIST_DIR=tmp_path / "chroma",
        INDEX_DIR=tmp_path / "indexes",
        LOG_DIR=tmp_path / "logs",
        MODEL_DIR=tmp_path / "models",
        METADATA_CSV_PATH=tmp_path / "metadata.csv",
        DEVICE="cpu",
    )
    result.create_directories()
    return result
