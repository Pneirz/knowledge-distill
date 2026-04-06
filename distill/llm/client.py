import json
import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Protocol

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class BaseLLMClient(Protocol):
    """Common interface implemented by all LLM backends."""

    model: str

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str: ...

    def complete_json(
        self,
        system: str,
        user: str,
        max_tokens: int = 4096,
    ) -> dict: ...

    def get_usage_stats(self) -> dict: ...


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


class ClaudeCodeClient:
    """LLM client that delegates to the `claude` CLI (Claude Code).

    No API key required — uses the authenticated Claude Code session.
    Drop-in replacement for LLMClient: identical public interface.
    """

    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        if not shutil.which("claude"):
            raise RuntimeError(
                "claude CLI not found in PATH. "
                "Claude Code must be installed and authenticated."
            )
        self.model = model
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    def _run(self, system: str, user: str) -> str:
        """Call `claude -p` and return the assistant text response."""
        cmd = [
            "claude", "-p",
            "--model", self.model,
            "--system-prompt", system,
            "--output-format", "json",
            "--no-session-persistence",
            user,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=180)
        if result.returncode != 0:
            raise RuntimeError(f"claude CLI error: {result.stderr.decode('utf-8', errors='replace')[:500]}")
        envelope = json.loads(result.stdout.decode("utf-8"))
        return envelope.get("result", "")

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        return self._run(system, user)

    def complete_json(
        self,
        system: str,
        user: str,
        max_tokens: int = 4096,
    ) -> dict:
        enhanced_user = (
            user + "\n\nIMPORTANT: Respond with ONLY valid JSON. No markdown fences, no explanation."
        )
        raw = self._run(system, enhanced_user).strip()
        # Strip markdown code fences if the model adds them anyway
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        raw = raw.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.error("claude CLI returned invalid JSON: %s", raw[:500])
            raise ValueError(f"claude CLI response is not valid JSON: {exc}") from exc

    def get_usage_stats(self) -> dict:
        return {
            "input_tokens": self._total_input_tokens,
            "output_tokens": self._total_output_tokens,
        }


class CodexCLIClient:
    """LLM client backed by the `codex exec` CLI.

    Uses the local Codex authentication/session and does not require API keys.
    """

    def __init__(self, model: str = "gpt-5") -> None:
        if not shutil.which("codex"):
            raise RuntimeError(
                "codex CLI not found in PATH. "
                "Codex CLI must be installed and authenticated."
            )
        self.model = model
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    def _run(self, prompt: str) -> str:
        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".txt", encoding="utf-8", delete=False
        ) as output_file:
            output_path = Path(output_file.name)

        cmd = [
            "codex",
            "exec",
            "--skip-git-repo-check",
            "--ephemeral",
            "--sandbox",
            "read-only",
            "--color",
            "never",
            "--model",
            self.model,
            "--output-last-message",
            str(output_path),
            prompt,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                stderr = result.stderr[:500] if result.stderr else "unknown error"
                raise RuntimeError(f"codex CLI error: {stderr}")
            return output_path.read_text(encoding="utf-8").strip()
        finally:
            output_path.unlink(missing_ok=True)

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        prompt = (
            "System instructions:\n"
            f"{system}\n\n"
            "User request:\n"
            f"{user}"
        )
        return self._run(prompt)

    def complete_json(
        self,
        system: str,
        user: str,
        max_tokens: int = 4096,
    ) -> dict:
        prompt = (
            "System instructions:\n"
            f"{system}\n\n"
            "User request:\n"
            f"{user}\n\n"
            "Return ONLY valid JSON. No markdown fences. No commentary."
        )
        raw = self._run(prompt).strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        raw = raw.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.error("codex CLI returned invalid JSON: %s", raw[:500])
            raise ValueError(f"codex CLI response is not valid JSON: {exc}") from exc

    def get_usage_stats(self) -> dict:
        return {
            "input_tokens": self._total_input_tokens,
            "output_tokens": self._total_output_tokens,
        }
