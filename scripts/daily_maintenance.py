"""Daily maintenance runner for incremental data refresh and housekeeping.

Examples:
  python scripts/daily_maintenance.py
  python scripts/daily_maintenance.py --days 2 --symbols NIFTY,BANKNIFTY --timeframes 5m,1d --strict-quality
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run daily maintenance workflow.")
    parser.add_argument("--settings", default="config/settings.yaml", help="Settings YAML path")
    parser.add_argument("--days", type=int, default=2, help="Incremental lookback window in days")
    parser.add_argument("--symbols", default="NIFTY,BANKNIFTY", help="Comma-separated symbols")
    parser.add_argument("--timeframes", default="1d,5m", help="Comma-separated timeframes")
    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        help="Skip preflight health check",
    )
    parser.add_argument(
        "--skip-quality-report",
        action="store_true",
        help="Skip candle quality report generation",
    )
    parser.add_argument(
        "--strict-quality",
        action="store_true",
        help="Fail workflow when quality thresholds are breached",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running remaining tasks after a task failure",
    )
    return parser.parse_args()


def run_cmd(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def render_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def main() -> int:
    args = parse_args()
    python_bin = sys.executable
    symbols = parse_csv(args.symbols)
    timeframes = parse_csv(args.timeframes)

    steps: list[tuple[str, list[str]]] = []
    if not args.skip_health_check:
        steps.append(
            (
                "health_check",
                [python_bin, "scripts/health_check.py", "--quick"],
            )
        )

    download_cmd = [
        python_bin,
        "scripts/download_historical.py",
        "--days",
        str(args.days),
        "--settings",
        args.settings,
    ]
    for symbol in symbols:
        download_cmd.extend(["--symbol", symbol])
    for timeframe in timeframes:
        download_cmd.extend(["--timeframe", timeframe])
    steps.append(("incremental_download", download_cmd))

    if not args.skip_quality_report:
        quality_cmd = [python_bin, "scripts/data_quality_report.py", "--settings", args.settings]
        if args.strict_quality:
            quality_cmd.append("--strict")
        steps.append(("quality_report", quality_cmd))

    started_at = datetime.now(UTC)
    print("=" * 72)
    print("NiftyQuant Daily Maintenance")
    print("=" * 72)
    print(f"started_at={started_at.isoformat()}")
    print(f"symbols={symbols}")
    print(f"timeframes={timeframes}")
    print(f"days={args.days}")
    print(f"strict_quality={args.strict_quality}")

    failures: list[str] = []
    for step_name, cmd in steps:
        print(f"\n[RUN] {step_name}: {render_cmd(cmd)}")
        code, out, err = run_cmd(cmd)
        if out.strip():
            print(out.rstrip())
        if err.strip():
            print(err.rstrip())

        if code == 0:
            print(f"[OK] {step_name}")
            continue

        print(f"[FAIL] {step_name} exit_code={code}")
        failures.append(step_name)
        if not args.continue_on_error:
            break

    ended_at = datetime.now(UTC)
    print("\n" + "=" * 72)
    print("Daily Maintenance Summary")
    print("=" * 72)
    print(f"ended_at={ended_at.isoformat()}")
    print(f"duration_seconds={(ended_at - started_at).total_seconds():.1f}")
    print(f"steps_total={len(steps)}")
    print(f"steps_failed={len(failures)}")
    if failures:
        print(f"failed_steps={','.join(failures)}")
        return 1
    print("failed_steps=none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
