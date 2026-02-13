"""Minimal Kite WebSocket stream helper.

Usage:
  python scripts/kite_ws_stream.py --tokens 738561,5633 --mode full
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional
    load_dotenv = None

if load_dotenv:
    load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream Kite ticks for instrument tokens.")
    parser.add_argument(
        "--tokens",
        required=True,
        help="Comma-separated instrument tokens (e.g. 738561,5633).",
    )
    parser.add_argument(
        "--mode",
        choices=["ltp", "quote", "full"],
        default="quote",
        help="Tick mode for subscribed tokens.",
    )
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=0,
        help="Stop after this many on_ticks callbacks. 0 = run until interrupted.",
    )
    return parser.parse_args()


def _parse_tokens(raw: str) -> list[int]:
    out: list[int] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        out.append(int(token))
    if not out:
        raise ValueError("No valid instrument tokens provided.")
    return out


def main() -> int:
    try:
        from kiteconnect import KiteTicker
    except ImportError:
        print("ERROR: kiteconnect package is not installed. Install with: pip install kiteconnect")
        return 1

    args = parse_args()
    api_key = (os.getenv("KITE_API_KEY") or "").strip()
    access_token = (os.getenv("KITE_ACCESS_TOKEN") or "").strip()
    if not api_key:
        print("ERROR: KITE_API_KEY is missing.")
        return 1
    if not access_token:
        print("ERROR: KITE_ACCESS_TOKEN is missing.")
        return 1

    tokens = _parse_tokens(args.tokens)
    mode_map = {
        "ltp": KiteTicker.MODE_LTP,
        "quote": KiteTicker.MODE_QUOTE,
        "full": KiteTicker.MODE_FULL,
    }
    selected_mode = mode_map[args.mode]

    kws = KiteTicker(api_key, access_token)
    tick_callbacks = 0
    stop_requested = False

    def _request_stop(_signum=None, _frame=None):
        nonlocal stop_requested
        stop_requested = True
        try:
            kws.stop()
        except Exception:
            pass

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    def on_ticks(ws, ticks):
        nonlocal tick_callbacks
        tick_callbacks += 1
        print(json.dumps({"event": "ticks", "count": len(ticks), "ticks": ticks}, default=str))
        if args.max_ticks > 0 and tick_callbacks >= args.max_ticks:
            ws.stop()

    def on_connect(ws, _response):
        print(f"connected; subscribing tokens={tokens} mode={args.mode}")
        ws.subscribe(tokens)
        ws.set_mode(selected_mode, tokens)

    def on_error(_ws, code, reason):
        print(f"ws_error code={code} reason={reason}")

    def on_close(_ws, code, reason):
        print(f"ws_closed code={code} reason={reason}")

    kws.on_ticks = on_ticks
    kws.on_connect = on_connect
    kws.on_error = on_error
    kws.on_close = on_close

    try:
        kws.connect()
    finally:
        if stop_requested:
            print("stopped by signal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

