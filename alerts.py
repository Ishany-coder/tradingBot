"""Alerting for the unattended trader: never fail silently.

Channels, all best-effort (an alert failure must never break a trading cycle):
  * macOS Notification Center (always available locally, zero setup)
  * optional webhook — set ALERT_WEBHOOK_URL in .env to a Discord webhook or
    Slack-compatible endpoint and every alert is POSTed there too
  * optional healthcheck ping — set HEALTHCHECK_URL (e.g. healthchecks.io) and
    each successful cycle pings it; a missed ping alerts you externally (a
    dead-man's switch for the whole loop)
"""
from __future__ import annotations

import subprocess

import requests

from broker import _load_env


def notify(title: str, message: str) -> None:
    """Send an alert everywhere configured. Never raises."""
    try:  # macOS Notification Center
        script = f'display notification "{message[:200]}" with title "{title[:60]}"'
        subprocess.run(["osascript", "-e", script], capture_output=True, timeout=10)
    except Exception:  # noqa: BLE001
        pass
    url = _load_env().get("ALERT_WEBHOOK_URL", "")
    if url:
        try:  # Discord-style {"content": ...}; Slack accepts {"text": ...} too
            requests.post(url, json={"content": f"**{title}**\n{message}",
                                     "text": f"{title}: {message}"}, timeout=10)
        except Exception:  # noqa: BLE001
            pass


def heartbeat() -> None:
    """Ping the dead-man's-switch URL (if configured). Never raises."""
    url = _load_env().get("HEALTHCHECK_URL", "")
    if url:
        try:
            requests.get(url, timeout=10)
        except Exception:  # noqa: BLE001
            pass
