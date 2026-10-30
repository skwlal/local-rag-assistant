"""Local RAG application."""

import os

# Runtime never downloads weights; preparation is a separate explicit command.
for _key, _value in {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "RAGAS_DO_NOT_TRACK": "true",
    "ANONYMIZED_TELEMETRY": "False",
    "TOKENIZERS_PARALLELISM": "false",
    "LANGCHAIN_TRACING_V2": "false",
    "LANGSMITH_TRACING": "false",
    "OTEL_SDK_DISABLED": "true",
}.items():
    os.environ[_key] = _value
