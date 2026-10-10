"""GPU window budget (offline, read-only): ``python -m slh window caps``.

``serve start --cap`` is CUMULATIVE for the serving session: admission
requires ``committed + reservation <= cap``, where ``committed`` is everything
the session's ledger already holds (charged plus pending). A per-window cap
therefore has to be passed as ``committed + this window's reservation``,
read from the ledger at the time, never a hardcoded figure.

The window's aggregate is its GPU reservation plus every token subcap (the
diagnostic run, each screen replay run, the proposer cap); it must not exceed
the approved window total. Nothing here calls Modal or any model; the ledger
is read, never locked or repaired.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent


def committed_usd(ledger_path: str | Path) -> float:
    """Charged plus pending spend of a ledger, replayed read-only."""
    from slh.ledger import read_records

    path = Path(ledger_path)
    if not path.is_file():
        return 0.0
    pending: dict[str, float] = {}
    charged = 0.0
    for record in read_records(path):
        if record.get("op") == "reserve":
            pending[record["id"]] = float(record["usd"])
        elif record.get("op") in ("settle", "charge"):
            pending.pop(record.get("id"), None)
            charged += float(record["usd"])
    return round(charged + sum(pending.values()), 6)


def reservation_usd(window_min: int, *, episode_margin_min: int,
                    prices: str | Path = REPO / "configs/prices.yaml") -> float:
    """Exactly what ``serve start`` reserves for this window (same plan code)."""
    from slh.serving import build_allowance_usd, load_rate, plan_window

    plan = plan_window(window_min * 60, load_rate(prices, "serve"), time.time(),
                       fixed_charge_usd=build_allowance_usd(prices),
                       episode_margin_s=episode_margin_min * 60)
    return float(plan["reservation_usd"])


def caps(*, session: str | Path, window_min: int, episode_margin_min: int,
         diagnostic_cap: float, screen_cap: float, screen_runs: int, proposer_cap: float,
         max_window_usd: float, prices: str | Path = REPO / "configs/prices.yaml",
         preflight_cap: float = 0.0) -> dict[str, Any]:
    """``preflight_cap``: the provider preflight's synthetic request bound
    (amendment 5), a token subcap of the window like the others."""
    for name, value in (("diagnostic_cap", diagnostic_cap), ("screen_cap", screen_cap),
                        ("proposer_cap", proposer_cap), ("max_window_usd", max_window_usd),
                        ("preflight_cap", preflight_cap)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite nonnegative number")
    committed = committed_usd(Path(session) / "ledger.jsonl")
    reservation = reservation_usd(window_min, episode_margin_min=episode_margin_min,
                                  prices=prices)
    # Rounded UP to the cent so admission (committed + reservation <= cap) holds.
    gpu_cap = math.ceil((committed + reservation) * 100 - 1e-9) / 100
    tokens = round(diagnostic_cap + screen_runs * screen_cap + proposer_cap + preflight_cap, 6)
    total = round(reservation + tokens, 6)
    return {"committed_usd": committed, "gpu_reservation_usd": round(reservation, 6),
            "gpu_cap_usd": gpu_cap, "token_caps_usd": tokens,
            "diagnostic_cap_usd": diagnostic_cap, "screen_cap_usd": screen_cap,
            "screen_runs": screen_runs, "proposer_cap_usd": proposer_cap,
            "preflight_cap_usd": preflight_cap,
            "window_total_usd": total, "max_window_usd": max_window_usd,
            "within_cap": total <= max_window_usd + 1e-9}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m slh window", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    cmd = commands.add_parser("caps", help="cumulative GPU cap and window total (exit 2 if "
                              "the window total exceeds --max-window-usd)")
    cmd.add_argument("--session", type=Path, default=REPO / "serving/smoke")
    cmd.add_argument("--window-min", type=int, required=True)
    cmd.add_argument("--episode-margin-min", type=int, required=True)
    cmd.add_argument("--diagnostic-cap", type=float, required=True)
    cmd.add_argument("--screen-cap", type=float, required=True)
    cmd.add_argument("--screen-runs", type=int, required=True)
    cmd.add_argument("--proposer-cap", type=float, required=True)
    cmd.add_argument("--max-window-usd", type=float, required=True)
    cmd.add_argument("--preflight-cap", type=float, default=0.0,
                     help="the provider preflight's synthetic request bound (amendment 5)")
    cmd.add_argument("--shell", action="store_true", help="print KEY=value lines")
    args = parser.parse_args(argv)
    try:
        result = caps(session=args.session, window_min=args.window_min,
                      episode_margin_min=args.episode_margin_min,
                      diagnostic_cap=args.diagnostic_cap, screen_cap=args.screen_cap,
                      screen_runs=args.screen_runs, proposer_cap=args.proposer_cap,
                      max_window_usd=args.max_window_usd, preflight_cap=args.preflight_cap)
    except (ValueError, OSError) as exc:
        print(f"window caps refused: {exc}")
        return 2
    if args.shell:
        print(f"GPU_CAP={result['gpu_cap_usd']:.2f}")
        print(f"GPU_RESERVATION={result['gpu_reservation_usd']:.6f}")
        print(f"COMMITTED={result['committed_usd']:.6f}")
        print(f"WINDOW_TOTAL={result['window_total_usd']:.6f}")
    else:
        print(json.dumps(result, sort_keys=True))
    if not result["within_cap"]:
        print(f"window total ${result['window_total_usd']:.6f} exceeds "
              f"${result['max_window_usd']:.2f}")
        return 2
    return 0
