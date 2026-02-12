"""Health check utility for environment, dependencies, and data/API connectivity."""

import argparse
import os
import sys
from datetime import date, datetime, timedelta
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
    return "✓" if success else "✗"


def check_env_vars(require_broker_env: bool = True) -> bool:
    """Verify required environment variables are set."""
    print("\n━━━ Environment Variables ━━━")
    required = {
        "TRUEDATA_USERNAME": "TrueData login",
        "TRUEDATA_PASSWORD": "TrueData login",
        "KITE_API_KEY": "Zerodha Kite Connect",
        "KITE_API_SECRET": "Zerodha Kite Connect",
    }
    optional = {
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
            print(f"  {check_mark(ok)} {var} ({desc}){'' if ok else ' — MISSING'}")
        else:
            marker = "○" if ok else "·"
            suffix = "" if ok else " — not set (skipped in local-only mode)"
            print(f"  {marker} {var} ({desc}){suffix}")

    for var, desc in optional.items():
        value = os.getenv(var)
        ok = value is not None and len(value) > 0
        print(f"  {'○' if ok else '·'} {var} ({desc}){'' if ok else ' — not set (optional)'}")

    mode = os.getenv("TRADING_MODE", "development")
    print(f"\n  Trading mode: {mode}")
    return all_ok


def check_truedata() -> bool:
    """Test TrueData credentials and SDK availability."""
    print("\n━━━ TrueData Connection ━━━")
    username = os.getenv("TRUEDATA_USERNAME")
    password = os.getenv("TRUEDATA_PASSWORD")
    if not username or not password:
        print("  ✗ TrueData credentials not configured")
        return False

    print(f"  ✓ Credentials found for user: {username[:3]}***")
    try:
        __import__("truedata_ws")
        print("  ✓ truedata_ws SDK import successful")
    except ImportError:
        print("  ○ truedata_ws SDK not installed; live login test skipped")
        print("    Install vendor SDK to enable full TrueData connectivity checks")
    except Exception as e:
        print(f"  ✗ TrueData SDK import failed: {e}")
        return False

    return True


def check_kite() -> bool:
    """Test Zerodha Kite Connect API and optional live profile check."""
    print("\n━━━ Zerodha Kite Connect ━━━")
    try:
        api_key = os.getenv("KITE_API_KEY")
        if not api_key:
            print("  ✗ Kite API key not configured")
            return False

        print(f"  ✓ API key found: {api_key[:4]}***")

        try:
            from kiteconnect import KiteConnect
        except ImportError:
            print("  ○ kiteconnect not installed; live connectivity test skipped")
            return False

        access_token = os.getenv("KITE_ACCESS_TOKEN")
        if not access_token:
            print("  ○ Access token not set — daily auth still required")
            print("    Run your token generation flow before market open")
            return False

        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        profile = kite.profile()
        user_id = profile.get("user_id", "unknown")
        user_name = profile.get("user_name", "unknown")
        print(f"  ✓ Live API check passed: {user_id} ({user_name})")
        return True

    except Exception as e:
        print(f"  ✗ Kite connection failed: {e}")
        return False


def check_free_data(quick: bool = False) -> bool:
    """Test free data sources as fallback."""
    print("\n━━━ Free Data Sources ━━━")
    results = []

    # jugaad-data
    try:
        from jugaad_data.nse import stock_df
        if quick:
            print("  ✓ jugaad-data: import check passed (quick mode)")
            results.append(True)
        else:
            df = stock_df(
                symbol="SBIN",
                from_date=date.today() - timedelta(days=7),
                to_date=date.today(),
                series="EQ",
            )
            rows = len(df)
            print(f"  ✓ jugaad-data: fetched {rows} rows for SBIN")
            results.append(True)
    except ImportError:
        print("  ○ jugaad-data: not installed (pip install jugaad-data)")
        results.append(False)
    except Exception as e:
        print(f"  ✗ jugaad-data: {e}")
        results.append(False)

    # yfinance
    try:
        import yfinance as yf
        if quick:
            print("  ✓ yfinance: import check passed (quick mode)")
            results.append(True)
        else:
            nifty = yf.Ticker("^NSEI")
            hist = nifty.history(period="5d")
            rows = len(hist)
            print(f"  ✓ yfinance: fetched {rows} rows for NIFTY 50")
            results.append(True)
    except ImportError:
        print("  ○ yfinance: not installed (pip install yfinance)")
        results.append(False)
    except Exception as e:
        print(f"  ✗ yfinance: {e}")
        results.append(False)

    return any(results)


def check_directories() -> bool:
    """Ensure required directories exist."""
    print("\n━━━ Directory Structure ━━━")
    dirs = [
        REPO_ROOT / "data" / "cache",
        REPO_ROOT / "data" / "logs",
        REPO_ROOT / "data" / "reports",
        REPO_ROOT / "config",
    ]
    all_ok = True
    for d in dirs:
        path = Path(d)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ Created: {path.relative_to(REPO_ROOT)}")
        else:
            print(f"  ✓ Exists: {path.relative_to(REPO_ROOT)}")
    return all_ok


def check_dependencies() -> bool:
    """Verify critical Python packages are importable."""
    print("\n━━━ Python Dependencies ━━━")
    packages = {
        "numpy": "Core computation",
        "pandas": "Data manipulation",
        "scipy": "Statistical functions",
        "yaml": "Config loading (pyyaml)",
        "dotenv": "Environment loading (python-dotenv)",
        "pydantic": "Data validation",
        "structlog": "Structured logging",
    }

    all_ok = True
    for pkg, desc in packages.items():
        try:
            __import__(pkg)
            print(f"  ✓ {pkg} ({desc})")
        except ImportError:
            print(f"  ✗ {pkg} ({desc}) — not installed")
            all_ok = False

    return all_ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NiftyQuant health checks.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run import-level checks for free data sources (skip network calls).",
    )
    parser.add_argument(
        "--skip-broker-checks",
        action="store_true",
        help="Skip TrueData/Kite connectivity checks.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 55)
    print("  NiftyQuant Health Check")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}")
    print("=" * 55)

    results = {}
    results["env"] = check_env_vars(require_broker_env=not args.skip_broker_checks)
    results["deps"] = check_dependencies()
    results["dirs"] = check_directories()
    if args.skip_broker_checks:
        print("\n━━━ Broker Checks ━━━")
        print("  ○ Skipped (use --skip-broker-checks only for local setup validation)")
        results["truedata"] = True
        results["kite"] = True
    else:
        results["truedata"] = check_truedata()
        results["kite"] = check_kite()
    results["free_data"] = check_free_data(quick=args.quick)

    # Summary
    print("\n" + "=" * 55)
    print("  Summary")
    print("=" * 55)
    for name, ok in results.items():
        print(f"  {check_mark(ok)} {name}")

    dev_ready = results["env"] and results["deps"] and results["dirs"]
    live_api_ready = results["kite"] and results["truedata"]

    if dev_ready and live_api_ready and not args.skip_broker_checks:
        print("\n  → Ready for development and live API integration.")
        return

    if dev_ready and args.skip_broker_checks:
        print("\n  → Ready for local development setup. Run without --skip-broker-checks for live API checks.")
        return

    if dev_ready and not live_api_ready:
        print("\n  → Dev environment is ready; complete broker/data auth before live checks pass.")
        return

    print("\n  → Fix critical issues above before continuing.")
    sys.exit(1)


if __name__ == "__main__":
    main()
