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


PIT_YEARS = 9  # window for the point-in-time universes (matches the dashboard)


def build_targets(force: bool = False, universe: str | None = None,
                  method: str | None = None, variant: str | None = None):
    """Run the single live strategy; return (label, date, target_book, ic).

    universe / method / variant default to the deployed config (LIVE_UNIVERSE,
    LIVE_METHOD, STRATEGY_VARIANT) so the trader and the recurring loop trade the
    SAME model. Only the requested variant is built (the other walk-forward is
    skipped), so a live run is ~half the time of the old dual-variant build.
    """
    universe = universe or C.LIVE_UNIVERSE
    method = method or C.LIVE_METHOD
    variant = variant or C.STRATEGY_VARIANT

    if universe == "sp500":
        import sp500
        uni, members = sp500.build_universe(PIT_YEARS, force=force)
        bundle = backtest.run_all(force=force, universe_override=uni,
                                  years=PIT_YEARS, method=method, variant=variant,
                                  membership=members, fundamentals_source="edgar")
    elif universe == "pit2020":
        import universe_2020
        bundle = backtest.run_all(force=force, method=method, variant=variant,
                                  universe_override=universe_2020.HOLDINGS_2020,
                                  years=PIT_YEARS)
    else:
        bundle = backtest.run_all(force=force, method=method, variant=variant)

    res = bundle.result_a if variant == "A" else bundle.result_b
    date, book = backtest.current_book(res)
    return res.label, date, book, res.ic


def plan_orders(book: dict, equity: float, positions: dict) -> list[dict]:
    """Diff target weights against live positions into a list of orders.

    Only emits an order when |target$ - current$| exceeds the rebalance band
    (anti-churn). Names held but no longer in the book are fully liquidated.

    Crash guard: buy-side REBALANCE orders are suppressed for names already held
    that are down more than 10% (unrealized) — the intra-cycle loop must not
    mechanically average into a single-name crash. Sells are never suppressed.
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
        if (delta > 0 and t in positions
                and positions[t].get("unrealized_plpc", 0.0) < -0.10):
            print(f"  [crash-guard] {t} down "
                  f"{positions[t]['unrealized_plpc']:.0%} — not buying the dip.")
            continue
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


_BOOK_CACHE = C.STATE_DIR / "current_book.json"
_HWM_FILE = C.STATE_DIR / "hwm.json"
_ERR_STREAK = C.STATE_DIR / "err_streak"


def _halted() -> str | None:
    if C.STOP_FILE.exists():
        return f"STOP file present ({C.STOP_FILE}); trading halted."
    if not C.ENABLED:
        return "config.ENABLED is False; trading halted."
    return None


def _check_drawdown_kill(equity: float) -> str | None:
    """High-water-mark drawdown kill switch: unattended systems need an
    automated stop, not just a manual STOP file. Writes STOP + alerts."""
    hwm = equity
    if _HWM_FILE.exists():
        try:
            hwm = max(float(json.loads(_HWM_FILE.read_text())["hwm"]), equity)
        except Exception:  # noqa: BLE001
            pass
    _HWM_FILE.write_text(json.dumps({"hwm": hwm}))
    if equity < hwm * (1.0 - C.MAX_LIVE_DRAWDOWN):
        reason = (f"equity ${equity:,.0f} is {1 - equity / hwm:.0%} below "
                  f"high-water-mark ${hwm:,.0f} (limit {C.MAX_LIVE_DRAWDOWN:.0%})")
        C.STOP_FILE.write_text(reason)
        return reason
    return None


def _order_error_streak(had_errors: bool) -> int:
    """Track consecutive cycles with order errors; STOP after the limit."""
    streak = 0
    if _ERR_STREAK.exists():
        try:
            streak = int(_ERR_STREAK.read_text().strip() or 0)
        except ValueError:
            streak = 0
    streak = streak + 1 if had_errors else 0
    _ERR_STREAK.write_text(str(streak))
    return streak


def _frozen_book(signal_month: str):
    """Return the stored (book, ic) if it belongs to ``signal_month``, else None.

    The signal is a MONTHLY decision: intra-month cycles must trade toward the
    same frozen target, not re-derive a drifting one from partial-month data
    (and they skip the ~20-min model rebuild entirely).
    """
    if not _BOOK_CACHE.exists():
        return None
    try:
        rec = json.loads(_BOOK_CACHE.read_text())
        if rec.get("month") == signal_month:
            return rec["label"], rec["date"], rec["book"], rec.get("ic", {})
    except Exception:  # noqa: BLE001
        pass
    return None


def _store_book(label, date, book, ic, signal_month: str):
    _BOOK_CACHE.write_text(json.dumps(
        {"month": signal_month, "label": label, "date": str(date),
         "book": book, "ic": ic}, default=str))


def run(force: bool = False, force_dry: bool = False, allow_closed: bool = False,
        universe: str | None = None, method: str | None = None,
        variant: str | None = None) -> dict:
    """Execute one trading cycle. Returns a summary dict (also logged)."""
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    universe = universe or C.LIVE_UNIVERSE
    method = method or C.LIVE_METHOD
    variant = variant or C.STRATEGY_VARIANT

    import alerts

    halt = _halted()
    if halt:
        print(f"[trader] {halt}")
        return {"time": now, "halted": halt, "orders": []}

    broker = PaperBroker()  # raises unless paper endpoint + creds present

    # Automated kill switch: drawdown vs high-water mark (before anything else).
    equity = broker.get_equity()
    kill = _check_drawdown_kill(equity)
    if kill:
        print(f"[trader] KILL: {kill}")
        alerts.notify("Trader HALTED (drawdown)", kill)
        return {"time": now, "halted": kill, "orders": []}

    # Monthly signal, frozen intra-month: only rebuild when the month changes.
    signal_month = dt.date.today().strftime("%Y-%m")
    frozen = None if force else _frozen_book(signal_month)
    new_signal = frozen is None
    if frozen:
        label, date, book, ic = frozen
        print(f"[trader] using frozen {signal_month} book "
              f"({len(book)} names) — drift correction only.")
    else:
        label, date, book, ic = build_targets(force=force, universe=universe,
                                              method=method, variant=variant)
        if book:
            _store_book(label, date, book, ic, signal_month)
    if not book:
        print("[trader] no target book (data too thin); nothing to do.")
        alerts.notify("Trader: empty book", "No target book this cycle "
                      "(data too thin / breadth floor). Holding prior positions.")
        return {"time": now, "note": "empty book", "orders": []}

    # Cancel stale open orders so the diff below is against reality, and a
    # crash mid-cycle can't leave orphans that double-fill later.
    market_open = broker.is_market_open()
    if market_open and not force_dry:
        try:
            n_cancelled = broker.cancel_all_orders()
            if n_cancelled:
                print(f"[trader] cancelled {n_cancelled} stale open orders.")
        except Exception as exc:  # noqa: BLE001
            print(f"[trader] cancel-open-orders failed (continuing): {exc}")

    positions = broker.get_positions()
    orders = plan_orders(book, equity, positions)

    # Turnover circuit breaker: a large planned reshuffle WITHOUT a new monthly
    # signal means bad data or drift gone wrong — don't trade it unattended.
    planned_turnover = sum(o["dollars"] for o in orders) / equity if equity else 0.0
    breaker = (planned_turnover > C.MAX_CYCLE_TURNOVER and not new_signal)
    if breaker:
        msg = (f"planned turnover {planned_turnover:.0%} > "
               f"{C.MAX_CYCLE_TURNOVER:.0%} without a new monthly signal — "
               "forcing dry-run (bad-data guard).")
        print(f"[trader] BREAKER: {msg}")
        alerts.notify("Trader circuit breaker", msg)
        force_dry = True

    # DRY-RUN ONCE: first ever run sends nothing, then arms live trading.
    first_run = not C.DRYRUN_FLAG.exists()
    dry = force_dry or first_run or not (market_open or allow_closed)
    reason = ("forced dry-run" if force_dry else
              "first run (dry-run once)" if first_run else
              "market closed" if not market_open else "live")

    # lambdarank's "pred" is an ordinal ranking score, not a return — label it
    # accordingly (gbm/mlp pred IS an estimated forward return).
    pred_is_return = method in ("gbm", "mlp")
    print(f"\n=== trader run @ {now} ===")
    print(f"universe={universe}  model={method}  variant={label}  "
          f"signal_date={date}  equity=${equity:,.2f}  market_open={market_open}  "
          f"mode={'DRY-RUN' if dry else 'LIVE'} ({reason})")
    if ic.get("n_months"):
        print(f"backtest IC (signal quality): mean {ic['mean_ic']:+.3f}  "
              f"IR {ic['ic_ir']:.2f}  over {ic['n_months']} months")
    print(f"target book ({len(book)} names):")
    for t in sorted(book, key=lambda x: -book[x]["weight"]):
        b = book[t]
        score = (f"est_ret={b['pred']:+.2%}" if pred_is_return
                 else f"rank_score={b['pred']:+.3f}")
        print(f"  {t:<6} w={b['weight']:6.2%}  conf={b['confidence']:.2f}  {score}")

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
                # Deterministic id: a retry of the same cycle/symbol/side is
                # rejected by Alpaca as a duplicate instead of double-filling.
                coid = f"tb-{now[:13]}-{o['symbol']}-{o['side']}"
                res = broker.submit_notional(o["symbol"], o["dollars"], o["side"],
                                             dry_run=dry, client_order_id=coid)
            placed.append({**o, "result": "dry-run" if dry else res.get("id", "ok")})
        except Exception as exc:  # noqa: BLE001 - keep going on a single bad order
            print(f"    ! order failed for {o['symbol']}: {exc}")
            placed.append({**o, "result": f"error: {exc}"})

    if first_run and not force_dry:
        C.DRYRUN_FLAG.write_text(now)
        print(f"\n[trader] dry-run complete. Live trading ARMED for next run "
              f"(flag: {C.DRYRUN_FLAG}).")

    # Count only EXECUTED notional (skip dry-run plans and failed orders) so the
    # log reflects real activity, not what we merely planned.
    traded_notional = sum(
        p["dollars"] for p in placed
        if p.get("result") not in ("dry-run", None)
        and not str(p.get("result", "")).startswith("error"))
    # Alerting: errors first (streak-based STOP), then a fill summary; then the
    # dead-man heartbeat so an external monitor notices if the loop dies.
    errors = [p for p in placed if str(p.get("result", "")).startswith("error")]
    streak = _order_error_streak(bool(errors))
    if errors:
        alerts.notify(f"Trader: {len(errors)} order error(s)",
                      "; ".join(f"{p['symbol']}: {p['result']}" for p in errors[:5]))
        if streak >= C.MAX_ORDER_ERROR_STREAK:
            reason_stop = (f"{streak} consecutive cycles with order errors — "
                           "auto-halting.")
            C.STOP_FILE.write_text(reason_stop)
            alerts.notify("Trader HALTED (error streak)", reason_stop)
    elif not dry and traded_notional > 0:
        alerts.notify("Trader: orders placed",
                      f"{len(placed) - len(errors)} orders, "
                      f"${traded_notional:,.0f} notional; equity ${equity:,.0f}.")
    alerts.heartbeat()

    summary = {"time": now, "universe": universe, "method": method,
               "variant": label, "signal_date": str(date), "equity": equity,
               "market_open": market_open, "mode": reason,
               "new_signal": new_signal,
               "ic_mean": ic.get("mean_ic"), "ic_ir": ic.get("ic_ir"),
               "n_names": len(book),
               "gross_exposure": sum(b["weight"] for b in book.values()),
               "planned_turnover": planned_turnover,
               "traded_notional": traded_notional,
               "turnover_pct": (traded_notional / equity) if equity else None,
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
    ap.add_argument("--universe", default=None,
                    choices=["current", "pit2020", "sp500"],
                    help="override universe (default: config.LIVE_UNIVERSE)")
    ap.add_argument("--model", default=None,
                    choices=["gbm", "lambdarank", "mlp"],
                    help="override model (default: config.LIVE_METHOD)")
    ap.add_argument("--variant", default=None, choices=["A", "B"],
                    help="override variant (default: config.STRATEGY_VARIANT)")
    args = ap.parse_args()
    run(force=args.force, force_dry=args.dry_run, allow_closed=args.allow_closed,
        universe=args.universe, method=args.model, variant=args.variant)


if __name__ == "__main__":
    main()
