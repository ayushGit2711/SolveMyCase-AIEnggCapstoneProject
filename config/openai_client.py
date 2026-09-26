"""Factory for OpenAI clients configured from Settings.

Every OpenAI client in the app is built here, so all calls share one timeout and retry policy. The SDK default
is a 600 s timeout with 2 retries, which could stall a request for many minutes before any fallback runs.
"""

import logging
from typing import Any, Optional

from solvemycase.config.settings import Settings

logger = logging.getLogger(__name__)


def openai_api_key_value(settings: Settings) -> str:
    """Return the configured OpenAI API key, or an empty string when none is set."""
    key = settings.openai_api_key
    return key.get_secret_value() if key else ""


def build_openai_client(
    settings: Settings,
    *,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
) -> Optional[Any]:
    """Build an ``openai.OpenAI`` client with bounded timeout and retries.

    Args:
        settings: Application settings (API key, default timeout and retries).
        timeout: Per-request timeout in seconds (defaults to ``settings.openai_timeout_seconds``).
        max_retries: Automatic retries (defaults to ``settings.openai_max_retries``).

    Returns:
        The client, or None when no API key is configured or the SDK is unavailable (callers then run offline).
    """
    api_key = openai_api_key_value(settings)
    if not api_key:
        return None
    try:
        from openai import OpenAI
    except ImportError as err:  # pragma: no cover - openai is a hard dependency
        logger.warning("OpenAI SDK unavailable (%s); running offline.", err)
        return None
    return OpenAI(
        api_key=api_key,
        timeout=settings.openai_timeout_seconds if timeout is None else timeout,
        max_retries=settings.openai_max_retries if max_retries is None else max_retries,
    )
