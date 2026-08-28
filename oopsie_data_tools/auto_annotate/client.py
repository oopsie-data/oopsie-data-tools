"""Chat client for the CostPlusIQ gateway.

The gateway is OpenAI-compatible, so this is a thin stdlib wrapper rather than a
dependency. Three behaviours of the served model drove the shape of it:

*It is a reasoning model.* Completions carry a ``reasoning_content`` field, and the model
spends real budget there before writing ``content``. A budget sized for the answer alone
returns ``finish_reason: "length"`` with an empty string — silently, and it looks exactly
like a refusal. ``max_tokens`` defaults high here, and an empty completion is raised as an
error rather than parsed as an empty annotation.

*It honours ``response_format: json_schema``.* Output is schema-constrained, so the parse
step is a plain ``json.loads`` and the enums do not need defensive normalisation.

*It is a shared endpoint.* Transient 429/5xx are retried with exponential backoff.

One deployment detail also matters: the gateway sits behind Cloudflare, which rejects
urllib's default ``Python-urllib/3.x`` agent with a 403 "error code: 1010" that looks
nothing like an auth or quota problem. Any ordinary agent string passes, so one is always
sent.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from oopsie_data_tools.auto_annotate import net

# 520-527 are Cloudflare's own edge errors (524 = origin timed out), which show up under
# load on long video prompts and are always worth retrying.
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 525, 527}

# Cloudflare 403s the default urllib agent. See the module docstring.
USER_AGENT = "oopsie-auto-annotate/1.0"


class ModelError(RuntimeError):
    pass


class Client:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: int = 300,
        max_retries: int = 4,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    @staticmethod
    def image_part(jpeg_b64: str) -> Dict[str, Any]:
        return {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64," + jpeg_b64},
        }

    @staticmethod
    def video_part(path) -> Dict[str, Any]:
        """A whole video as a data URI.

        The gateway accepts ``video_url`` only; a ``{"type": "video"}`` part is rejected
        outright by request validation. Normalise the clip with :mod:`clip` before calling
        this -- source-resolution video bills an order of magnitude more.
        """
        import base64
        from pathlib import Path as _Path

        encoded = base64.b64encode(_Path(path).read_bytes()).decode("ascii")
        return {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + encoded}}

    @staticmethod
    def text_part(text: str) -> Dict[str, Any]:
        return {"type": "text", "text": text}

    def complete(
        self,
        messages: List[Dict[str, Any]],
        json_schema: Optional[Dict[str, Any]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        """One chat completion. Returns ``{content, reasoning, usage, raw}``."""
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_schema is not None:
            payload["response_format"] = {"type": "json_schema", "json_schema": json_schema}

        body = json.dumps(payload).encode()
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=body,
            headers={
                "Authorization": "Bearer " + self.api_key,
                "Content-Type": "application/json",
                "User-Agent": USER_AGENT,
            },
            method="POST",
        )

        last_error: Optional[str] = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout, context=net.context()
                ) as response:
                    data = json.load(response)
                break
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", "replace")[:400]
                last_error = f"HTTP {error.code}: {detail}"
                if error.code not in RETRY_STATUS or attempt == self.max_retries:
                    raise ModelError(last_error) from None
            except (urllib.error.URLError, TimeoutError) as error:
                last_error = str(error)
                if attempt == self.max_retries:
                    raise ModelError(last_error) from None
            time.sleep(min(2 ** attempt, 16))
        else:  # pragma: no cover - loop always breaks or raises
            raise ModelError(last_error or "exhausted retries")

        choice = data["choices"][0]
        message = choice.get("message", {})
        content = (message.get("content") or "").strip()
        if not content:
            raise ModelError(
                f"empty completion (finish_reason={choice.get('finish_reason')!r}); "
                "the reasoning budget likely consumed max_tokens"
            )
        return {
            "content": content,
            "reasoning": message.get("reasoning_content") or "",
            "usage": data.get("usage", {}),
            "raw": data,
        }

    def complete_json(
        self, messages, json_schema, max_tokens: int = 8000, growth: int = 2, **kwargs
    ) -> Dict[str, Any]:
        """``complete`` plus a parse, growing the budget if reasoning exhausts it.

        How much this model thinks before answering varies a lot with how ambiguous the
        episode is, so a budget that suits most episodes still truncates a few. Rather than
        sizing for the worst case on every call, the budget is doubled once and retried --
        only the episodes that need the headroom pay for it.
        """
        try:
            result = self.complete(
                messages, json_schema=json_schema, max_tokens=max_tokens, **kwargs
            )
        except ModelError as error:
            if "empty completion" not in str(error):
                raise
            result = self.complete(
                messages, json_schema=json_schema, max_tokens=max_tokens * growth, **kwargs
            )
        try:
            result["parsed"] = json.loads(result["content"])
        except json.JSONDecodeError as error:
            raise ModelError(f"model returned non-JSON: {error}: {result['content'][:300]}")
        return result
