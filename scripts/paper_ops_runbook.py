"""M6 paper-ready operations runbook (open/intraday/close)."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.store import DataStore
from src.ops.gates import (
    build_default_intraday_gates,
    build_default_open_gates,
    evaluate_freshness_gates,
    summarize_gate_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper ops runbook for M6.")
    parser.add_argument("--phase", choices=["open", "intraday", "close"], required=True)
    parser.add_argument("--symbol", default="NIFTY")
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--settings", default="config/settings.yaml")
    parser.add_argument("--output-dir", default="data/reports/ops")
    parser.add_argument("--strict", action="store_true", help="Fail on warning gates too.")
    parser.add_argument(
        "--run-health-check",
        action="store_true",
        help="Run scripts/health_check.py --quick before gates.",
    )
    parser.add_argument(
        "--run-maintenance",
        action="store_true",
        help="For close phase, run scripts/daily_maintenance.py.",
    )
    parser.add_argument(
        "--run-vectorbt",
        action="store_true",
        help="For close phase, run walk-forward vectorbt report.",
    )
    parser.add_argument(
        "--run-paper-fills-report",
        action="store_true",
        help="For close phase, run paper fills summary report.",
    )
    parser.add_argument(
        "--send-telegram",
        action="store_true",
        help="Send summary alert using TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID.",
    )
    return parser.parse_args()


def load_settings(path: str) -> dict:
    p = REPO_ROOT / path
    if not p.exists():
        raise FileNotFoundError(f"Settings file not found: {p}")
    return yaml.safe_load(p.read_text())


def run_cmd(cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "cmd": " ".join(shlex.quote(part) for part in cmd),
        "exit_code": int(proc.returncode),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "ok": proc.returncode == 0,
    }


def maybe_send_telegram(*, enabled: bool, summary: dict[str, Any], report_path: Path) -> None:
    if not enabled:
        return
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return
    phase = summary.get("phase", "unknown")
    ok = bool(summary.get("ok", False))
    hard = int(summary.get("hard_failures", 0))
    warn = int(summary.get("warning_failures", 0))
    icon = "OK" if ok else "FAIL"
    text = (
        f"NiftyQuant M6 {icon}\\n"
        f"phase={phase}\\n"
        f"hard_failures={hard} warning_failures={warn}\\n"
        f"report={report_path}"
    )
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=10)
    except requests.RequestException:
        return


def _vectorbt_cmd(*, args: argparse.Namespace, settings: dict) -> list[str]:
    backtest_cfg = settings.get("backtest", {})
    start = str(backtest_cfg.get("start_date", "2022-01-01"))
    end = str(backtest_cfg.get("end_date", datetime.now(UTC).strftime("%Y-%m-%d")))
    return [
        sys.executable,
        "scripts/run_vectorbt_research.py",
        "--symbol",
        args.symbol,
        "--timeframe",
        args.timeframe,
        "--walk-forward",
        "--from",
        start,
        "--to",
        end,
    ]


def main() -> int:
    args = parse_args()
    settings = load_settings(args.settings)
    cache_dir = REPO_ROOT / settings.get("data", {}).get("cache_dir", "data/cache")
    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC).replace(tzinfo=None)

    store = DataStore(base_dir=str(cache_dir))
    steps: list[dict[str, Any]] = []

    if args.run_health_check:
        steps.append(run_cmd([sys.executable, "scripts/health_check.py", "--quick"]))

    if args.phase == "open":
        gates = build_default_open_gates(settings, symbol=args.symbol, timeframe=args.timeframe)
    elif args.phase == "intraday":
        gates = build_default_intraday_gates(settings, symbol=args.symbol, timeframe=args.timeframe)
    else:
        gates = []

    gate_results = evaluate_freshness_gates(store, gates, now=now) if gates else []
    gate_summary = summarize_gate_results(gate_results) if gate_results else {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "hard_failures": 0,
        "warning_failures": 0,
        "ok": True,
        "results": [],
    }

    if args.phase == "close":
        if args.run_maintenance:
            steps.append(run_cmd([sys.executable, "scripts/daily_maintenance.py"]))
        if args.run_vectorbt:
            steps.append(run_cmd(_vectorbt_cmd(args=args, settings=settings)))
        if args.run_paper_fills_report:
            steps.append(
                run_cmd(
                    [
                        sys.executable,
                        "scripts/paper_fills_report.py",
                        "--symbol",
                        args.symbol,
                    ]
                )
            )

    steps_ok = all(step.get("ok", False) for step in steps)
    hard_failures = int(gate_summary.get("hard_failures", 0))
    warning_failures = int(gate_summary.get("warning_failures", 0))
    ok = steps_ok and hard_failures == 0 and (warning_failures == 0 or not args.strict)

    payload: dict[str, Any] = {
        "generated_at": now.isoformat(),
        "phase": args.phase,
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "strict": bool(args.strict),
        "ok": bool(ok),
        "steps_ok": bool(steps_ok),
        "hard_failures": hard_failures,
        "warning_failures": warning_failures,
        "gate_summary": gate_summary,
        "steps": steps,
    }

    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    report_path = out_dir / f"paper_ops_{args.phase}_{args.symbol}_{args.timeframe}_{stamp}.json"
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    print("=" * 72)
    print("Paper Ops Runbook")
    print("=" * 72)
    print(f"phase={args.phase} symbol={args.symbol} timeframe={args.timeframe}")
    print(
        f"gate_total={gate_summary['total']} passed={gate_summary['passed']} "
        f"hard_failures={hard_failures} warning_failures={warning_failures}"
    )
    if steps:
        for i, step in enumerate(steps, start=1):
            print(f"step[{i}] ok={step['ok']} exit_code={step['exit_code']} cmd={step['cmd']}")
    print(f"ok={ok}")
    print(f"report={report_path.relative_to(REPO_ROOT)}")

    maybe_send_telegram(enabled=args.send_telegram, summary=payload, report_path=report_path)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
