"""Health check utility for environment, dependencies, and Kite connectivity."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv:
    load_dotenv()


def check_mark(success: bool) -> str:
    return "[OK]" if success else "[X]"


def check_env_vars(require_broker_env: bool = True) -> bool:
    """Verify required environment variables are set."""
    print("\n=== Environment Variables ===")
    required = {
        "KITE_API_KEY": "Zerodha Kite Connect",
        "KITE_API_SECRET": "Zerodha Kite Connect",
    }
    optional = {
        "KITE_REDIRECT_URL": "OAuth redirect URL",
        "KITE_ACCESS_TOKEN": "Daily session token",
        "TELEGRAM_BOT_TOKEN": "Telegram alerts",
        "TELEGRAM_CHAT_ID": "Telegram alerts",
        "TRADING_MODE": "Trading mode (development/paper/live)",
    }

    all_ok = True
    for var, desc in required.items():
        value = os.getenv(var)
        ok = value is not None and len(value) > 0
        if not ok and require_broker_env:
            all_ok = False
        if require_broker_env:
            suffix = "" if ok else " - MISSING"
            print(f"  {check_mark(ok)} {var} ({desc}){suffix}")
        else:
            marker = "[~]" if ok else "[ ]"
            suffix = "" if ok else " - not set (skipped in local-only mode)"
            print(f"  {marker} {var} ({desc}){suffix}")

    for var, desc in optional.items():
        value = os.getenv(var)
        ok = value is not None and len(value) > 0
        marker = "[~]" if ok else "[ ]"
        suffix = "" if ok else " - not set (optional)"
        print(f"  {marker} {var} ({desc}){suffix}")

    mode = os.getenv("TRADING_MODE", "development")
    print(f"\n  Trading mode: {mode}")
    return all_ok


def check_kite(quick: bool = False) -> bool:
    """Test Zerodha Kite Connect API and optional quote smoke checks."""
    print("\n=== Zerodha Kite Connect ===")
    try:
        api_key = os.getenv("KITE_API_KEY")
        if not api_key:
            print("  [X] Kite API key not configured")
            return False

        print(f"  [OK] API key found: {api_key[:4]}***")

        try:
            from kiteconnect import KiteConnect
        except ImportError:
            print("  [X] kiteconnect not installed. Install with: pip install kiteconnect")
            return False

        access_token = os.getenv("KITE_ACCESS_TOKEN")
        if not access_token:
            print("  [X] Access token not set - run kite auth flow first")
            return False

        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)

        profile = kite.profile()
        user_id = profile.get("user_id", "unknown")
        user_name = profile.get("user_name", "unknown")
        print(f"  [OK] Auth/profile check passed: {user_id} ({user_name})")

        if quick:
            print("  [~] Quick mode: skipping live quote smoke checks")
            return True

        symbols = ["NSE:NIFTY 50", "NSE:INDIA VIX"]
        quotes = kite.quote(symbols)
        bad = []
        for symbol in symbols:
            ltp = float(quotes.get(symbol, {}).get("last_price", 0.0) or 0.0)
            if ltp <= 0:
                bad.append(symbol)

        if bad:
            print(f"  [X] Quote smoke check failed for: {', '.join(bad)}")
            return False

        print("  [OK] Live quote smoke check passed (NIFTY + INDIA VIX)")
        return True

    except Exception as exc:
        print(f"  [X] Kite connection failed: {exc}")
        return False


def check_directories() -> bool:
    """Ensure required directories exist."""
    print("\n=== Directory Structure ===")
    dirs = [
        REPO_ROOT / "data" / "cache",
        REPO_ROOT / "data" / "logs",
        REPO_ROOT / "data" / "reports",
        REPO_ROOT / "config",
    ]
    for directory in dirs:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"  [OK] Created: {path.relative_to(REPO_ROOT)}")
        else:
            print(f"  [OK] Exists: {path.relative_to(REPO_ROOT)}")
    return True


def check_dependencies() -> bool:
    """Verify critical Python packages are importable."""
    print("\n=== Python Dependencies ===")
    packages = {
        "numpy": "Core computation",
        "pandas": "Data manipulation",
        "scipy": "Statistical functions",
        "yaml": "Config loading (pyyaml)",
        "dotenv": "Environment loading (python-dotenv)",
        "pydantic": "Data validation",
        "structlog": "Structured logging",
        "kiteconnect": "Broker/data API",
    }

    all_ok = True
    for pkg, desc in packages.items():
        try:
            __import__(pkg)
            print(f"  [OK] {pkg} ({desc})")
        except ImportError:
            print(f"  [X] {pkg} ({desc}) - not installed")
            all_ok = False

    return all_ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NiftyQuant health checks.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run auth/profile check only (skip live quote smoke checks).",
    )
    parser.add_argument(
        "--skip-broker-checks",
        action="store_true",
        help="Skip Kite connectivity checks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 55)
    print("  NiftyQuant Health Check")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}")
    print("=" * 55)

    results: dict[str, bool] = {}
    results["env"] = check_env_vars(require_broker_env=not args.skip_broker_checks)
    results["deps"] = check_dependencies()
    results["dirs"] = check_directories()

    if args.skip_broker_checks:
        print("\n=== Broker Checks ===")
        print("  [~] Skipped (use --skip-broker-checks only for local setup validation)")
        results["kite"] = True
    else:
        results["kite"] = check_kite(quick=args.quick)

    print("\n" + "=" * 55)
    print("  Summary")
    print("=" * 55)
    for name, ok in results.items():
        print(f"  {check_mark(ok)} {name}")

    dev_ready = results["env"] and results["deps"] and results["dirs"]
    live_api_ready = results["kite"]

    if dev_ready and live_api_ready and not args.skip_broker_checks:
        print("\n  -> Ready for development and live API integration.")
        return

    if dev_ready and args.skip_broker_checks:
        print("\n  -> Ready for local development setup. Run without --skip-broker-checks for live API checks.")
        return

    if dev_ready and not live_api_ready:
        print("\n  -> Dev environment is ready; complete Kite auth/connectivity before live checks pass.")
        return

    print("\n  -> Fix critical issues above before continuing.")
    sys.exit(1)


if __name__ == "__main__":
    main()
