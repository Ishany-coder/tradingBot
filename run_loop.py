"""Unattended loop: recompute + trade every RECOMPUTE_HOURS, forever.

Each cycle calls ``trader.run()`` which is self-guarding (paper-only, kill-switch,
dry-run-once, rebalance band, market-hours gate). Leave this running, or wrap it
with launchd/cron (see README) for restart-on-boot.

  python run_loop.py
"""

from __future__ import annotations

import time

import config as C
import trader


def main():
    interval = C.RECOMPUTE_HOURS * 3600
    print(f"[loop] started — recompute every {C.RECOMPUTE_HOURS}h. "
          f"Create {C.STOP_FILE} to halt. Ctrl-C to stop the loop.")
    while True:
        try:
            trader.run()
        except Exception as exc:  # noqa: BLE001 - never let the loop die
            print(f"[loop] run error (continuing): {exc}")
        time.sleep(interval)


if __name__ == "__main__":
    main()
