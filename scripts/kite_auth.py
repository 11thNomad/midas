"""Kite Connect auth helper: login URL and request_token exchange.

Usage:
  python scripts/kite_auth.py login-url
  python scripts/kite_auth.py exchange --request-token <token>
  python scripts/kite_auth.py exchange --request-token <token> --print-env
"""

from __future__ import annotations

import argparse
import os
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
    parser = argparse.ArgumentParser(description="Kite Connect auth helper.")
    sub = parser.add_subparsers(dest="command", required=True)

    login_url = sub.add_parser("login-url", help="Print Kite login URL.")
    login_url.add_argument("--api-key", help="Kite API key. Defaults to KITE_API_KEY env.")

    exchange = sub.add_parser("exchange", help="Exchange request_token for access_token.")
    exchange.add_argument(
        "--request-token", required=True, help="request_token from redirect callback URL."
    )
    exchange.add_argument("--api-key", help="Kite API key. Defaults to KITE_API_KEY env.")
    exchange.add_argument("--api-secret", help="Kite API secret. Defaults to KITE_API_SECRET env.")
    exchange.add_argument(
        "--print-env",
        action="store_true",
        help="Print export line for KITE_ACCESS_TOKEN after successful exchange.",
    )
    return parser.parse_args()


def _require(value: str | None, label: str) -> str:
    if value and value.strip():
        return value.strip()
    raise ValueError(f"{label} is required.")


def _build_login_url(api_key: str) -> str:
    return f"https://kite.zerodha.com/connect/login?v=3&api_key={api_key}"


def _run_login_url(args: argparse.Namespace):
    api_key = _require(args.api_key or os.getenv("KITE_API_KEY"), "KITE_API_KEY")
    print(_build_login_url(api_key))


def _run_exchange(args: argparse.Namespace):
    api_key = _require(args.api_key or os.getenv("KITE_API_KEY"), "KITE_API_KEY")
    api_secret = _require(args.api_secret or os.getenv("KITE_API_SECRET"), "KITE_API_SECRET")
    request_token = _require(args.request_token, "request_token")

    try:
        from kiteconnect import KiteConnect
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError(
            "kiteconnect package is not installed. Install with: pip install kiteconnect"
        ) from exc

    kite = KiteConnect(api_key=api_key)
    data = kite.generate_session(request_token, api_secret=api_secret)
    access_token = str(data.get("access_token", "")).strip()
    if not access_token:
        raise RuntimeError("Kite did not return access_token.")

    user_id = str(data.get("user_id", "unknown"))
    print(f"Session generated for user_id={user_id}")
    print(f"access_token={access_token}")
    if args.print_env:
        print(f"export KITE_ACCESS_TOKEN='{access_token}'")


def main() -> int:
    args = parse_args()
    try:
        if args.command == "login-url":
            _run_login_url(args)
            return 0
        if args.command == "exchange":
            _run_exchange(args)
            return 0
        raise ValueError(f"Unknown command '{args.command}'")
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
