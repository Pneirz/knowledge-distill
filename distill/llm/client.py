import json
import logging

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class LLMClient:
    """Wrapper around the Anthropic client with retry logic and token tracking.

    State is justified: the client holds the Anthropic connection, the
    configured model name, and cumulative token usage for cost monitoring.
    """

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6") -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        """Send a single-turn prompt and return the assistant's text response."""
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        self._total_input_tokens += response.usage.input_tokens
        self._total_output_tokens += response.usage.output_tokens
        return response.content[0].text

    def complete_json(
        self,
        system: str,
        user: str,
        max_tokens: int = 4096,
    ) -> dict:
        """Request a JSON response and parse it before returning.

        Uses assistant prefill with '{' to steer the model toward valid JSON
        output, then parses and validates before returning.
        Raises ValueError if the response cannot be parsed as JSON.
        """
        # Prefill the assistant turn with '{' to guarantee JSON output
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0.0,
            system=system,
            messages=[
                {"role": "user", "content": user},
                {"role": "assistant", "content": "{"},
            ],
        )
        self._total_input_tokens += response.usage.input_tokens
        self._total_output_tokens += response.usage.output_tokens

        # Reconstruct full JSON string (the prefill '{' was consumed by the model)
        raw = "{" + response.content[0].text
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.error("LLM returned invalid JSON: %s", raw[:500])
            raise ValueError(f"LLM response is not valid JSON: {exc}") from exc

    def get_usage_stats(self) -> dict:
        """Return cumulative token usage for this session."""
        return {
            "input_tokens": self._total_input_tokens,
            "output_tokens": self._total_output_tokens,
        }
