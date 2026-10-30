from types import SimpleNamespace

import pytest

import chat


@pytest.mark.parametrize("commands", [["/model installed", "/exit"], []])
def test_chat_closes_and_does_not_mutate_default_model(settings, monkeypatch, commands):
    closed = []
    pipeline = SimpleNamespace(
        settings=settings,
        models=lambda: SimpleNamespace(ollama_models=[settings.DEFAULT_LLM_MODEL, "installed"]),
        close=lambda: closed.append(True),
    )
    initial = settings.DEFAULT_LLM_MODEL
    commands = iter(commands)

    def read_input(prompt):
        value = next(commands, None)
        if value is None:
            raise EOFError
        return value

    monkeypatch.setattr(chat, "LocalRAGPipeline", lambda settings: pipeline)
    monkeypatch.setattr("builtins.input", read_input)
    assert chat.main(["--model", initial]) == 0
    assert pipeline.settings.DEFAULT_LLM_MODEL == initial
    assert closed == [True]


def test_chat_closes_when_selected_model_is_missing(settings, monkeypatch):
    closed = []
    monkeypatch.setattr(
        chat,
        "LocalRAGPipeline",
        lambda settings: SimpleNamespace(
            models=lambda: SimpleNamespace(ollama_models=[]),
            close=lambda: closed.append(True),
        ),
    )
    with pytest.raises(ValueError, match="не установлена"):
        chat.main(["--model", "missing"])
    assert closed == [True]
