"""
Health Check — verify all API connections and data freshness.

Run this FIRST after setting up your .env file.
It validates that every external dependency is reachable and working.

Usage:
    python scripts/health_check.py
"""

import sys
import os
from datetime import datetime, date, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()


def check_mark(success: bool) -> str:
    return "✓" if success else "✗"


def check_env_vars() -> bool:
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
        if not ok:
            all_ok = False
        print(f"  {check_mark(ok)} {var} ({desc}){'' if ok else ' — MISSING'}")

    for var, desc in optional.items():
        value = os.getenv(var)
        ok = value is not None and len(value) > 0
        print(f"  {'○' if ok else '·'} {var} ({desc}){'' if ok else ' — not set (optional)'}")

    mode = os.getenv("TRADING_MODE", "development")
    print(f"\n  Trading mode: {mode}")
    return all_ok


def check_truedata() -> bool:
    """Test TrueData API connection."""
    print("\n━━━ TrueData Connection ━━━")
    try:
        # Note: actual TrueData import will depend on their SDK
        # This is a placeholder — replace with real connection test
        username = os.getenv("TRUEDATA_USERNAME")
        if not username:
            print("  ✗ TrueData credentials not configured")
            return False

        print(f"  ✓ Credentials found for user: {username[:3]}***")

        # TODO: Replace with actual TrueData connection test
        # from truedata_ws.TrueDataWebSocket import TrueDataWebSocket
        # td = TrueDataWebSocket(username, password)
        # td.disconnect()
        print("  ○ Live connection test — uncomment after installing TrueData SDK")
        return True

    except Exception as e:
        print(f"  ✗ TrueData connection failed: {e}")
        return False


def check_kite() -> bool:
    """Test Zerodha Kite Connect API."""
    print("\n━━━ Zerodha Kite Connect ━━━")
    try:
        api_key = os.getenv("KITE_API_KEY")
        if not api_key:
            print("  ✗ Kite API key not configured")
            return False

        print(f"  ✓ API key found: {api_key[:4]}***")

        access_token = os.getenv("KITE_ACCESS_TOKEN")
        if not access_token:
            print("  ○ Access token not set — needs daily login flow")
            print("    Run: python scripts/kite_auth.py to generate token")
        else:
            print(f"  ✓ Access token present")

        # TODO: Replace with actual Kite connection test
        # from kiteconnect import KiteConnect
        # kite = KiteConnect(api_key=api_key)
        # kite.set_access_token(access_token)
        # profile = kite.profile()
        # print(f"  ✓ Connected as: {profile['user_name']}")
        print("  ○ Live connection test — uncomment after installing kiteconnect")
        return True

    except Exception as e:
        print(f"  ✗ Kite connection failed: {e}")
        return False


def check_free_data() -> bool:
    """Test free data sources as fallback."""
    print("\n━━━ Free Data Sources ━━━")
    results = []

    # jugaad-data
    try:
        from jugaad_data.nse import stock_df
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
        "data/cache",
        "data/logs",
        "data/reports",
    ]
    all_ok = True
    for d in dirs:
        path = Path(d)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ Created: {d}")
        else:
            print(f"  ✓ Exists: {d}")
    return all_ok


def check_dependencies() -> bool:
    """Verify critical Python packages are importable."""
    print("\n━━━ Python Dependencies ━━━")
    packages = {
        "numpy": "Core computation",
        "pandas": "Data manipulation",
        "scipy": "Statistical functions",
        "yaml": "Config loading (pyyaml)",
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


def main():
    print("=" * 55)
    print("  NiftyQuant Health Check")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}")
    print("=" * 55)

    results = {}
    results["env"] = check_env_vars()
    results["deps"] = check_dependencies()
    results["dirs"] = check_directories()
    results["truedata"] = check_truedata()
    results["kite"] = check_kite()
    results["free_data"] = check_free_data()

    # Summary
    print("\n" + "=" * 55)
    print("  Summary")
    print("=" * 55)
    for name, ok in results.items():
        print(f"  {check_mark(ok)} {name}")

    critical_ok = results["env"] and results["deps"]
    if critical_ok:
        print("\n  → Ready for development. Set up API credentials to proceed.")
    else:
        print("\n  → Fix critical issues above before continuing.")
        sys.exit(1)


if __name__ == "__main__":
    main()
