"""Alpaca PAPER trading client (REST via requests).

Hard safety: every call goes through ``assert_paper()``, which refuses any
endpoint whose URL does not contain ``paper-api``. There is no live-money path.

Credentials come from ``.env`` (ALPACA_API_KEY / ALPACA_API_SECRET /
ALPACA_BASE_URL). Orders are NOTIONAL (fractional dollar) market orders so we
can size precisely to conviction weights.
"""

from __future__ import annotations

import os
from pathlib import Path

import requests

import config as C


def _load_env() -> dict[str, str]:
    """Minimal .env loader (no external dependency)."""
    env = {}
    path = Path(__file__).parent / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    # process env overrides file
    for k in ("ALPACA_API_KEY", "ALPACA_API_SECRET", "ALPACA_BASE_URL"):
        if os.environ.get(k):
            env[k] = os.environ[k]
    return env


class PaperBroker:
    """Read account/positions and submit notional orders on the paper account."""

    def __init__(self):
        env = _load_env()
        self.key = env.get("ALPACA_API_KEY", "")
        self.secret = env.get("ALPACA_API_SECRET", "")
        self.base = env.get("ALPACA_BASE_URL", "").rstrip("/")
        self.assert_paper()
        self.h = {"APCA-API-KEY-ID": self.key, "APCA-API-SECRET-KEY": self.secret}

    # --- safety -------------------------------------------------------------
    def assert_paper(self):
        """Refuse to operate against anything but the paper endpoint."""
        if C.PAPER_HOST_MARKER not in self.base:
            raise RuntimeError(
                f"REFUSING TO TRADE: ALPACA_BASE_URL ('{self.base}') is not a "
                f"paper endpoint (must contain '{C.PAPER_HOST_MARKER}'). "
                "This bot is paper-only by design."
            )
        if not self.key or not self.secret:
            raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_API_SECRET in .env")

    # --- reads --------------------------------------------------------------
    def get_account(self) -> dict:
        r = requests.get(f"{self.base}/account", headers=self.h, timeout=30)
        r.raise_for_status()
        return r.json()

    def get_equity(self) -> float:
        return float(self.get_account()["equity"])

    def get_positions(self) -> dict[str, dict]:
        """{ticker: {qty, market_value}} for current holdings."""
        r = requests.get(f"{self.base}/positions", headers=self.h, timeout=30)
        r.raise_for_status()
        return {p["symbol"]: {"qty": float(p["qty"]),
                              "market_value": float(p["market_value"]),
                              "unrealized_plpc": float(p.get("unrealized_plpc") or 0.0)}
                for p in r.json()}

    def is_market_open(self) -> bool:
        r = requests.get(f"{self.base}/clock", headers=self.h, timeout=30)
        r.raise_for_status()
        return bool(r.json().get("is_open", False))

    def get_open_orders(self) -> list[dict]:
        r = requests.get(f"{self.base}/orders", headers=self.h,
                         params={"status": "open", "limit": 500}, timeout=30)
        r.raise_for_status()
        return r.json()

    # --- writes -------------------------------------------------------------
    def cancel_all_orders(self) -> int:
        """Cancel every open order. Called at cycle start so the planner never
        diffs against positions while stale orders are still working (and a
        crash mid-cycle can't leave orphans that double-fill next cycle)."""
        r = requests.delete(f"{self.base}/orders", headers=self.h, timeout=30)
        r.raise_for_status()
        return len(r.json()) if isinstance(r.json(), list) else 0

    def submit_notional(self, symbol: str, dollars: float, side: str,
                        dry_run: bool = False,
                        client_order_id: str | None = None) -> dict:
        """Submit a fractional-$ market order. side = 'buy' | 'sell'.

        client_order_id (deterministic per cycle+symbol) makes retries
        idempotent: Alpaca rejects a duplicate id instead of double-filling.
        Returns the order JSON (or a synthetic record when dry_run).
        """
        dollars = round(abs(dollars), 2)
        plan = {"symbol": symbol, "notional": dollars, "side": side,
                "type": "market", "time_in_force": "day"}
        if client_order_id:
            plan["client_order_id"] = client_order_id[:48]
        if dry_run:
            return {"dry_run": True, **plan}
        r = requests.post(f"{self.base}/orders", headers=self.h, json=plan, timeout=30)
        r.raise_for_status()
        return r.json()

    def liquidate(self, symbol: str, dry_run: bool = False) -> dict:
        """Close an entire position (used when a name drops out of the book).

        cancel_orders=true: Alpaca 422s a position-close while the symbol has
        working orders unless they are cancelled with it.
        """
        if dry_run:
            return {"dry_run": True, "symbol": symbol, "action": "liquidate"}
        r = requests.delete(f"{self.base}/positions/{symbol}",
                            headers=self.h, params={"cancel_orders": "true"},
                            timeout=30)
        r.raise_for_status()
        return r.json()
