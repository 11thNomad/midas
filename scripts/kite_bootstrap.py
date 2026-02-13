"""One-stop Kite auth bootstrap utility.

This script helps you:
1) Print login URL
2) Exchange request_token
3) Persist KITE_ACCESS_TOKEN into .env
4) Optionally verify token by calling kite.profile()

Examples:
  python scripts/kite_bootstrap.py login-url
  python scripts/kite_bootstrap.py exchange --request-token <token> --save-env
  python scripts/kite_bootstrap.py exchange --request-token <token> --save-env --verify
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
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

if load_dotenv:
    load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kite auth bootstrap utility.")
    sub = parser.add_subparsers(dest="command", required=True)

    login_url = sub.add_parser("login-url", help="Print Kite login URL.")
    login_url.add_argument("--api-key", help="KITE_API_KEY override.")

    exchange = sub.add_parser("exchange", help="Exchange request token for access token.")
    exchange.add_argument("--request-token", required=True, help="request_token from redirect URL.")
    exchange.add_argument("--api-key", help="KITE_API_KEY override.")
    exchange.add_argument("--api-secret", help="KITE_API_SECRET override.")
    exchange.add_argument(
        "--save-env", action="store_true", help="Persist KITE_ACCESS_TOKEN to .env file."
    )
    exchange.add_argument("--env-file", default=".env", help="Path to env file (default: .env).")
    exchange.add_argument(
        "--verify", action="store_true", help="Call kite.profile() to verify token."
    )
    exchange.add_argument(
        "--print-env",
        action="store_true",
        help="Print export command for shell usage.",
    )
    return parser.parse_args()


def _require(value: str | None, label: str) -> str:
    if value and value.strip():
        return value.strip()
    raise ValueError(f"{label} is required.")


def _kite_connect(api_key: str):
    try:
        from kiteconnect import KiteConnect
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "kiteconnect package is not installed. Install with: pip install kiteconnect"
        ) from exc
    return KiteConnect(api_key=api_key)


def _login_url(api_key: str) -> str:
    return f"https://kite.zerodha.com/connect/login?v=3&api_key={api_key}"


def _upsert_env_value(env_path: Path, key: str, value: str):
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []

    replaced = False
    out: list[str] = []
    prefix = f"{key}="
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(prefix):
            out.append(f"{key}={value}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.append(f"{key}={value}")
    env_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def cmd_login_url(args: argparse.Namespace):
    api_key = _require(args.api_key or os.getenv("KITE_API_KEY"), "KITE_API_KEY")
    print("Open this URL in browser and complete login:")
    print(_login_url(api_key))


def cmd_exchange(args: argparse.Namespace):
    api_key = _require(args.api_key or os.getenv("KITE_API_KEY"), "KITE_API_KEY")
    api_secret = _require(args.api_secret or os.getenv("KITE_API_SECRET"), "KITE_API_SECRET")
    request_token = _require(args.request_token, "request_token")

    kite = _kite_connect(api_key)
    data = kite.generate_session(request_token, api_secret=api_secret)
    access_token = str(data.get("access_token", "")).strip()
    if not access_token:
        raise RuntimeError("Kite did not return access_token.")

    user_id = str(data.get("user_id", "unknown"))
    print(f"Session generated for user_id={user_id}")
    print(f"access_token={access_token}")

    if args.save_env:
        env_path = (REPO_ROOT / args.env_file).resolve()
        _upsert_env_value(env_path, "KITE_ACCESS_TOKEN", access_token)
        print(f"Saved KITE_ACCESS_TOKEN to {env_path}")

    if args.print_env:
        print(f"export KITE_ACCESS_TOKEN='{access_token}'")

    if args.verify:
        kite.set_access_token(access_token)
        profile = kite.profile()
        uid = str(profile.get("user_id", "unknown"))
        uname = str(profile.get("user_name", "unknown"))
        print(f"Verification OK: user_id={uid} user_name={uname}")


def main() -> int:
    args = parse_args()
    try:
        if args.command == "login-url":
            cmd_login_url(args)
            return 0
        if args.command == "exchange":
            cmd_exchange(args)
            return 0
        raise ValueError(f"Unknown command '{args.command}'")
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
