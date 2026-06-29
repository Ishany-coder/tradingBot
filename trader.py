"""Automated paper trader — one run = compute targets, reconcile, place orders.

Flow per run:
  1. honour kill-switch (STOP file) and ENABLED flag; broker asserts paper-only
  2. run the strategy pipeline -> latest conviction target weights
  3. read live equity + positions from Alpaca
  4. diff to target $; trade only names whose drift exceeds the rebalance band
  5. DRY-RUN ONCE: the very first run prints the plan and sends nothing, then
     writes a flag; every run after is fully automatic (no human input)
  6. append a JSON record of the run to the trade log

CLI:
  python trader.py            # one automatic run
  python trader.py --force    # re-download data first
  python trader.py --dry-run  # force a no-send run (plan only)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json

import config as C
import backtest
from broker import PaperBroker


def build_targets(force: bool = False):
    """Run the strategy and return (variant_label, date, target_book).

    Prefers Backtest B (quality-screened); falls back to A if B is data-starved.
    """
    bundle = backtest.run_all(force=force)
    res = bundle.result_b if bundle.result_b.holdings_history else bundle.result_a
    date, book = backtest.current_book(res)
    return res.label, date, book


def plan_orders(book: dict, equity: float, positions: dict) -> list[dict]:
    """Diff target weights against live positions into a list of orders.

    Only emits an order when |target$ - current$| exceeds the rebalance band
    (anti-churn). Names held but no longer in the book are fully liquidated.
    """
    invest = equity * C.INVEST_FRACTION
    targets = {t: book[t]["weight"] * invest for t in book}
    current = {t: positions[t]["market_value"] for t in positions}
    band = max(C.REBALANCE_BAND * equity, C.MIN_ORDER_USD)

    orders: list[dict] = []

    # Exit names that fell out of the book.
    for t, mv in current.items():
        if t not in targets and mv > 0:
            orders.append({"symbol": t, "side": "sell", "action": "liquidate",
                           "dollars": mv, "target_w": 0.0})

    # Enter / adjust names in the target book.
    for t, tgt in targets.items():
        cur = current.get(t, 0.0)
        delta = tgt - cur
        if abs(delta) < band:
            continue  # within band -> leave it alone
        orders.append({
            "symbol": t,
            "side": "buy" if delta > 0 else "sell",
            "action": "rebalance",
            "dollars": abs(delta),
            "target_w": book[t]["weight"],
            "confidence": book[t]["confidence"],
            "pred": book[t]["pred"],
        })
    return orders


def _halted() -> str | None:
    if C.STOP_FILE.exists():
        return f"STOP file present ({C.STOP_FILE}); trading halted."
    if not C.ENABLED:
        return "config.ENABLED is False; trading halted."
    return None


def run(force: bool = False, force_dry: bool = False, allow_closed: bool = False) -> dict:
    """Execute one trading cycle. Returns a summary dict (also logged)."""
    now = dt.datetime.now(dt.timezone.utc).isoformat()

    halt = _halted()
    if halt:
        print(f"[trader] {halt}")
        return {"time": now, "halted": halt, "orders": []}

    broker = PaperBroker()  # raises unless paper endpoint + creds present

    label, date, book = build_targets(force=force)
    if not book:
        print("[trader] no target book (data too thin); nothing to do.")
        return {"time": now, "note": "empty book", "orders": []}

    equity = broker.get_equity()
    positions = broker.get_positions()
    orders = plan_orders(book, equity, positions)

    # DRY-RUN ONCE: first ever run sends nothing, then arms live trading.
    first_run = not C.DRYRUN_FLAG.exists()
    market_open = broker.is_market_open()
    dry = force_dry or first_run or not (market_open or allow_closed)
    reason = ("forced dry-run" if force_dry else
              "first run (dry-run once)" if first_run else
              "market closed" if not market_open else "live")

    print(f"\n=== trader run @ {now} ===")
    print(f"variant={label}  signal_date={date}  equity=${equity:,.2f}  "
          f"market_open={market_open}  mode={'DRY-RUN' if dry else 'LIVE'} ({reason})")
    print(f"target book ({len(book)} names):")
    for t in sorted(book, key=lambda x: -book[x]["weight"]):
        b = book[t]
        print(f"  {t:<6} w={b['weight']:6.2%}  conf={b['confidence']:.2f}  "
              f"est_ret={b['pred']:+.2%}")

    placed = []
    if not orders:
        print("no orders — all positions within the rebalance band.")
    for o in orders:
        print(f"  {o['action'].upper():<10} {o['side'].upper():<4} {o['symbol']:<6} "
              f"${o['dollars']:>10,.2f}" + ("   [DRY-RUN]" if dry else ""))
        try:
            if o["action"] == "liquidate":
                res = broker.liquidate(o["symbol"], dry_run=dry)
            else:
                res = broker.submit_notional(o["symbol"], o["dollars"], o["side"], dry_run=dry)
            placed.append({**o, "result": "dry-run" if dry else res.get("id", "ok")})
        except Exception as exc:  # noqa: BLE001 - keep going on a single bad order
            print(f"    ! order failed for {o['symbol']}: {exc}")
            placed.append({**o, "result": f"error: {exc}"})

    if first_run and not force_dry:
        C.DRYRUN_FLAG.write_text(now)
        print(f"\n[trader] dry-run complete. Live trading ARMED for next run "
              f"(flag: {C.DRYRUN_FLAG}).")

    summary = {"time": now, "variant": label, "signal_date": str(date),
               "equity": equity, "market_open": market_open, "mode": reason,
               "book": book, "orders": placed}
    with open(C.TRADE_LOG, "a") as fh:
        fh.write(json.dumps(summary, default=str) + "\n")
    return summary


def main():
    ap = argparse.ArgumentParser(description="Automated Alpaca paper trader")
    ap.add_argument("--force", action="store_true", help="re-download data first")
    ap.add_argument("--dry-run", action="store_true", help="plan only, send nothing")
    ap.add_argument("--allow-closed", action="store_true",
                    help="attempt to send even when market closed (Alpaca may reject)")
    args = ap.parse_args()
    run(force=args.force, force_dry=args.dry_run, allow_closed=args.allow_closed)


if __name__ == "__main__":
    main()
