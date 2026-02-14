"""Helpers for loading vectorbt parameter-set report artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError, ParserError


def list_vectorbt_paramset_runs(
    root_dir: Path,
    *,
    prefix: str = "vectorbt_paramsets_",
) -> list[Path]:
    if not root_dir.exists() or not root_dir.is_dir():
        return []
    out = [path for path in root_dir.iterdir() if path.is_dir() and path.name.startswith(prefix)]
    return sorted(out, key=lambda p: p.name, reverse=True)


def load_run_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "vectorbt_parameter_set_summary.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return raw if isinstance(raw, dict) else {}


def load_run_table(run_dir: Path, name: str) -> pd.DataFrame:
    path = run_dir / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_walkforward_folds(
    run_dir: Path,
    *,
    set_ids: list[str],
    fee_profiles: list[str],
) -> pd.DataFrame:
    return load_detail_artifact(
        run_dir,
        set_ids=set_ids,
        fee_profiles=fee_profiles,
        suffix="walkforward_folds",
    )


def load_detail_artifact(
    run_dir: Path,
    *,
    set_ids: list[str],
    fee_profiles: list[str],
    suffix: str,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    details_dir = run_dir / "details"
    roots: list[Path] = []
    if details_dir.exists() and details_dir.is_dir():
        roots.append(details_dir)
    roots.append(run_dir)

    seen_paths: set[Path] = set()
    for root in roots:
        for csv_path in sorted(root.glob(f"*_{suffix}.csv")):
            resolved = csv_path.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            parsed = parse_artifact_filename(
                csv_path=csv_path,
                set_ids=set_ids,
                fee_profiles=fee_profiles,
                suffix=suffix,
            )
            if parsed is None:
                continue
            set_id, fee_profile = parsed
            try:
                frame = pd.read_csv(csv_path)
            except (EmptyDataError, ParserError):
                continue
            if frame.empty:
                continue
            frame["set_id"] = set_id
            frame["fee_profile"] = fee_profile
            frame["source_file"] = csv_path.name
            rows.append(frame)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    if "fold" in out.columns:
        out["fold"] = pd.to_numeric(out["fold"], errors="coerce")
    return out


def parse_fold_filename(
    *,
    csv_path: Path,
    set_ids: list[str],
    fee_profiles: list[str],
) -> tuple[str, str] | None:
    return parse_artifact_filename(
        csv_path=csv_path,
        set_ids=set_ids,
        fee_profiles=fee_profiles,
        suffix="walkforward_folds",
    )


def parse_artifact_filename(
    *,
    csv_path: Path,
    set_ids: list[str],
    fee_profiles: list[str],
    suffix: str,
) -> tuple[str, str] | None:
    stem = csv_path.stem
    suffix_token = f"_{suffix}"
    if not stem.endswith(suffix_token):
        return None
    base = stem[: -len(suffix_token)]

    # Prefer longest set_id matches to avoid ambiguous prefixes.
    for set_id in sorted(set_ids, key=len, reverse=True):
        prefix = f"{set_id}_"
        if not base.startswith(prefix):
            continue
        fee_profile = base[len(prefix) :]
        if fee_profile in fee_profiles:
            return set_id, fee_profile

    # Fallback parser for unexpected names.
    if "_" not in base:
        return None
    set_id, fee_profile = base.rsplit("_", 1)
    return set_id, fee_profile
