# %%
import json
from unittest.mock import MagicMock, patch

import pytest

from distill.llm.client import LLMClient


@pytest.fixture
def mock_anthropic_response():
    """Build a fake Anthropic Messages API response."""
    def _make(text: str, input_tokens: int = 10, output_tokens: int = 20):
        response = MagicMock()
        response.content = [MagicMock(text=text)]
        response.usage.input_tokens = input_tokens
        response.usage.output_tokens = output_tokens
        return response
    return _make


@pytest.fixture
def client_with_mock(mock_anthropic_response):
    """LLMClient with the Anthropic SDK patched out."""
    with patch("distill.llm.client.anthropic.Anthropic") as mock_anthropic_cls:
        mock_instance = MagicMock()
        mock_anthropic_cls.return_value = mock_instance
        mock_instance.messages.create.return_value = mock_anthropic_response(
            "Hello from mock."
        )
        client = LLMClient(api_key="test-key", model="test-model")
        yield client, mock_instance


def test_complete_returns_text(client_with_mock):
    """complete() returns the text content of the first message block."""
    client, _ = client_with_mock
    result = client.complete(system="sys", user="user")
    assert result == "Hello from mock."


def test_complete_accumulates_token_usage(client_with_mock, mock_anthropic_response):
    """Token counts accumulate across multiple calls."""
    client, mock_instance = client_with_mock
    mock_instance.messages.create.return_value = mock_anthropic_response(
        "response", input_tokens=5, output_tokens=10
    )

    client.complete(system="s", user="u")
    client.complete(system="s", user="u")

    stats = client.get_usage_stats()
    assert stats["input_tokens"] == 10
    assert stats["output_tokens"] == 20


def test_complete_json_parses_valid_json(client_with_mock, mock_anthropic_response):
    """complete_json() parses the assistant response as JSON correctly.

    The prefill '{' is prepended; the mock returns only the rest of the JSON.
    """
    client, mock_instance = client_with_mock
    # The LLM response continues from the '{' prefill
    mock_instance.messages.create.return_value = mock_anthropic_response(
        '"key": "value"}'
    )
    result = client.complete_json(system="s", user="u")
    assert result == {"key": "value"}


def test_complete_json_raises_on_invalid_json(client_with_mock, mock_anthropic_response):
    """complete_json() raises ValueError when the response is not valid JSON."""
    client, mock_instance = client_with_mock
    mock_instance.messages.create.return_value = mock_anthropic_response(
        "this is not json}"
    )
    with pytest.raises(ValueError, match="not valid JSON"):
        client.complete_json(system="s", user="u")


def test_get_usage_stats_initial_zero():
    """Usage stats start at zero before any API calls."""
    with patch("distill.llm.client.anthropic.Anthropic"):
        client = LLMClient(api_key="test")
    stats = client.get_usage_stats()
    assert stats["input_tokens"] == 0
    assert stats["output_tokens"] == 0
