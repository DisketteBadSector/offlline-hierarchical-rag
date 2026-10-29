"""Shared helpers for offline model benchmark scripts (Phase 2).

Provides YAML loading, HTTP helpers, prompt assembler construction, and default paths
used by ``benchmark_offline_models.py`` and the unit test suite.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

try:
    import requests as _requests  # type: ignore[import-not-found]
except ImportError:
    _requests = None

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from offline_rag.domain.model_prompt import PromptAssembler  # noqa: E402
from offline_rag.infrastructure.config import (  # noqa: E402
    YamlPromptTemplateRepository,
    load_yaml_dict,
)

DEFAULT_MODELS_YAML = REPO_ROOT / "data" / "fixtures" / "models.yaml"
DEFAULT_PROMPTS_YAML = REPO_ROOT / "data" / "prompts"


def load_yaml_file(path: Path) -> dict[str, Any]:
    """Load a YAML file using the infrastructure YAML loader."""
    return load_yaml_dict(path)


def _http_get_json(url: str, timeout: float = 3.0) -> dict[str, Any]:
    """Perform HTTP GET and parse JSON using requests or urllib."""
    if _requests is not None:
        resp = _requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_post_json(url: str, payload: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
    """Perform HTTP POST with JSON payload using requests or urllib."""
    if _requests is not None:
        resp = _requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def build_prompt_assembler(prompts_path: Path = DEFAULT_PROMPTS_YAML) -> PromptAssembler:
    """Create a PromptAssembler backed by YamlPromptTemplateRepository."""
    repo = YamlPromptTemplateRepository(prompts_path=prompts_path)
    return PromptAssembler(repository=repo)


__all__ = [
    "DEFAULT_MODELS_YAML",
    "DEFAULT_PROMPTS_YAML",
    "REPO_ROOT",
    "_http_get_json",
    "_http_post_json",
    "build_prompt_assembler",
    "load_yaml_file",
]