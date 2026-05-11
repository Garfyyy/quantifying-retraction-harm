"""Matched-pair retraction analysis (harm panel).

Supported input parquet formats:
    A) Current vector-based format (your distance=1 file):
         - corpusid (match id), treated, publicationdate, RetractionDate, <outcome vector>
         - outcome vector length is typically 11; use --vector-offset to map vector to years 1..10

    B) Wide paper-level format:
         - paper_id, match_id, treated, pub_year, distance, retract_year, cites_y1..cites_y10

    C) Long/panel format:
         - paper_id (optional), match_id, treated, distance, age, y/outcome
         - plus either post_ret OR (year and retract_year) OR (pub_year and retract_year)
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def _require_scipy_stats():
    """Return scipy.stats or raise.

    This script is reviewer-facing; for exact p-values and critical values we require SciPy
    instead of using approximations.
    """
    stats = _try_scipy_stats()
    if stats is None:
        raise ImportError("SciPy is required for exact inference. Install via: pip install scipy")
    return stats


def _chi2_p_value(x: float, df: int) -> tuple[float | None, bool]:
    """Chi-square upper-tail p-value for statistic x with df degrees of freedom.

    Returns (p, used_approx). used_approx is always False; this script requires SciPy
    for exact inference rather than using approximations.
    """
    if df <= 0:
        return None, False
    stats = _require_scipy_stats()
    p = float(stats.chi2.sf(float(x), df=int(df)))
    return p, False


def _format_chi2_p(p: float | None, used_approx: bool) -> str:
    if p is None or not np.isfinite(p):
        return "NA"
    p = _clip_p01(float(p))
    if p == 0.0:
        return "<1e-300"
    return f"{p:.3e}"


def _try_scipy_stats():
    try:
        from scipy import stats  # type: ignore

        return stats
    except Exception:
        return None


def _parse_distances(s: str | None) -> list[int] | None:
    if s is None or s.strip() == "" or s.strip().lower() in {"all", "*"}:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out: list[int] = []
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            out.extend(list(range(int(a), int(b) + 1)))
        else:
            out.append(int(p))
    return sorted(set(out))


def _to_year(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="raise")
    return dt.dt.year.astype(int)


def _ensure_1to1_matches(df: pd.DataFrame, match_col: str = "match_id") -> None:
    # Each match must contain exactly 2 rows, one treated and one control
    sizes = df.groupby(match_col).size()
    bad_size = sizes[sizes != 2]
    if len(bad_size) > 0:
        raise ValueError(
            f"Each {match_col} must contain exactly 2 rows (1 treated, 1 control). "
            f"Found {len(bad_size)} problematic matches. Example: {bad_size.head(5).to_dict()}"
        )
    # Each match must contain exactly one treated=1
    treated_sums = df.groupby(match_col)["treated"].sum()
    bad = treated_sums[treated_sums != 1]
    if len(bad) > 0:
        raise ValueError(
            f"Each {match_col} must contain exactly one treated=1 and one treated=0. "
            f"Found {len(bad)} problematic matches. Example: {bad.head(5).to_dict()}"
        )


def _inherit_within_match(df: pd.DataFrame, cols: list[str], match_col: str = "match_id") -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            inherited = out.groupby(match_col)[c].transform("max")
            out[c] = out[c].fillna(inherited)
    return out


def _detect_format(df: pd.DataFrame, outcome_col: str) -> str:
    cols = set(df.columns)
    if {"corpusid", "treated", "publicationdate", "RetractionDate", outcome_col}.issubset(cols):
        return "vector_current"

    has_wide_cites = any(re.match(r"^cites_y\d+$", str(c)) for c in df.columns)
    if has_wide_cites and {"match_id", "treated", "pub_year", "distance"}.issubset(cols):
        return "wide_paper"

    if {"match_id", "treated", "age"}.issubset(cols):
        return "long_panel"

    return "unknown"


def _parquet_column_names(path: str) -> list[str]:
    try:
        import pyarrow.parquet as pq  # type: ignore

        pf = pq.ParquetFile(path)
        # Use Arrow schema to get correct top-level names for list columns.
        return list(pf.schema_arrow.names)
    except Exception:
        # Fallback: let pandas read full file if pyarrow metadata is unavailable.
        return []


def _infer_distance_from_path(path: str) -> int | None:
    m = re.search(r"(?:^|/)data_(\d+)\.parquet$", path)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _sample_match_ids_vector_current_parquet(
    path: str,
    *,
    n_matches: int,
    seed: int,
) -> set[int]:
    """Reservoir-sample match_ids from treated==1 rows (one per match in 1:1 design)."""
    if n_matches <= 0:
        return set()
    import pyarrow.parquet as pq  # type: ignore

    rng = np.random.default_rng(int(seed))
    pf = pq.ParquetFile(path)
    sample: list[int] = []
    seen = 0
    for rg in range(pf.num_row_groups):
        tab = pf.read_row_group(rg, columns=["corpusid", "treated"])
        df = tab.to_pandas()
        # treated==1 appears once per match in your 1:1 data.
        treated = df["treated"].to_numpy(dtype=int)
        ids = df["corpusid"].to_numpy()
        for mid in ids[treated == 1]:
            try:
                mid_int = int(mid)
            except Exception:
                continue
            seen += 1
            if len(sample) < n_matches:
                sample.append(mid_int)
            else:
                j = int(rng.integers(0, seen))
                if j < n_matches:
                    sample[j] = mid_int
    return set(sample)


def _load_raw_vector_current_subset(
    path: str,
    *,
    match_ids: set[int],
    outcome_col: str,
) -> pd.DataFrame:
    """Load a subset of raw vector_current rows whose corpusid is in match_ids."""
    if not match_ids:
        return pd.DataFrame()
    import pyarrow.parquet as pq  # type: ignore

    pf = pq.ParquetFile(path)
    cols = ["corpusid", "treated", "publicationdate", "RetractionDate", outcome_col]
    parts: list[pd.DataFrame] = []
    for rg in range(pf.num_row_groups):
        tab = pf.read_row_group(rg, columns=cols)
        df = tab.to_pandas()
        # membership filter on corpusid
        try:
            mids = df["corpusid"].astype(int)
            mask = mids.isin(match_ids)
        except Exception:
            mask = df["corpusid"].apply(lambda x: int(x) in match_ids if pd.notna(x) else False)
        sub = df.loc[mask]
        if len(sub) > 0:
            parts.append(sub)
    if not parts:
        return pd.DataFrame(columns=cols)
    return pd.concat(parts, ignore_index=True)


def _dy_event_study_on_panel(
    panel: pd.DataFrame,
    *,
    leads: int,
    lags: int,
) -> pd.DataFrame:
    """Compute dy(tau)-dy(-1) event-study on an in-memory long panel.

    dy(tau) is treated-control outcome difference within match_id at a given calendar year.
    Output columns: tau, est, se, t, p, ci_low, ci_high, n_clusters.
    """
    if panel.empty:
        return pd.DataFrame(columns=["tau", "est", "se", "t", "p", "ci_low", "ci_high", "n_clusters"])

    df = panel[["match_id", "treated", "year", "retract_year", "y"]].copy()
    df["tau"] = df["year"].astype(int) - df["retract_year"].astype(int)
    # pivot within match/year to get dy
    wide = (
        df.pivot_table(index=["match_id", "tau"], columns="treated", values="y", aggfunc="first")
        .rename(columns={0: "y_c", 1: "y_t"})
        .reset_index()
    )
    wide["dy"] = wide["y_t"] - wide["y_c"]

    # Build per-match reference dy(-1)
    ref = wide[wide["tau"] == -1][["match_id", "dy"]].rename(columns={"dy": "dy_ref"})
    merged = wide.merge(ref, on="match_id", how="inner")
    merged["dydelta"] = merged["dy"] - merged["dy_ref"]

    stats = _require_scipy_stats()

    out_rows = []
    for k in range(-int(leads), int(lags) + 1):
        if k == -1:
            continue
        sub = merged[merged["tau"] == int(k)]
        G = int(sub["match_id"].nunique())
        if G < 5:
            continue
        vals = sub["dydelta"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        est = float(np.mean(vals))
        # cluster = match_id, and here one obs per cluster per tau (after pivot), so SE is sd/sqrt(G)
        # Use ddof=1 for sample sd.
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
        se = sd / math.sqrt(G) if np.isfinite(sd) and G > 0 else float("nan")
        t = est / se if se and np.isfinite(se) else float("nan")
        df0 = max(int(G) - 1, 1)
        if np.isfinite(t):
            p = float(2.0 * stats.t.sf(abs(float(t)), df=df0))
        else:
            p = float("nan")
        if np.isfinite(se):
            tcrit = float(stats.t.ppf(0.975, df=df0))
            lo = est - tcrit * se
            hi = est + tcrit * se
        else:
            lo = float("nan")
            hi = float("nan")
        out_rows.append(
            {
                "tau": int(k),
                "est": est,
                "se": se,
                "t": t,
                "p": p,
                "ci_low": lo,
                "ci_high": hi,
                "n_clusters": G,
            }
        )
    return pd.DataFrame(out_rows).sort_values("tau")


def _namespace_ids(panel: pd.DataFrame, distance: int, source: str) -> pd.DataFrame:
    out = panel.copy()
    prefix = f"d{int(distance)}_"
    out["match_id"] = prefix + out["match_id"].astype(str)
    out["paper_id"] = prefix + out["paper_id"].astype(str)
    out["source_file"] = source
    return out


def _paper_id_from_match_treated(match_id: pd.Series, treated: pd.Series) -> pd.Series:
    """Create a collision-free paper_id from 1:1 matched pair identifiers.

    For integer match_id and treated in {0,1}, the mapping (match_id, treated) -> 2*match_id + treated
    is injective (no collisions). We additionally guard against int64 overflow.

    If match_id cannot be safely represented as int64, falls back to a string key.
    """

    t = treated.astype("int64")
    if not t.isin([0, 1]).all():
        raise ValueError("treated must be 0/1 to build paper_id")

    # Try numeric construction first (memory-friendly).
    try:
        m = match_id.astype("int64")
        max_safe = (np.iinfo(np.int64).max - 1) // 2
        if m.max() > max_safe or m.min() < 0:
            raise OverflowError("match_id out of safe range for int64 paper_id")
        return (m * 2 + t).astype("int64")
    except Exception:
        # Fallback: still collision-free, but more memory than int64.
        return match_id.astype(str) + "_" + treated.astype(str)

def _paper_id_vector_current(match_id: pd.Series, treated: pd.Series, distance: pd.Series) -> pd.Series:
    """Paper identifier for vector_current when corpusid is the treated-paper ID.

    In the provided vector_current layout, `match_id` equals the treated-paper identifier and is reused
    across distances, while the matched control can differ by distance. For pooled (cross-distance)
    regressions with paper fixed effects, we must avoid falsely treating different controls (same treated id,
    different distance) as the same paper.

    Design:
      - treated==1: stable across distances (same treated paper_id)
      - treated==0: distance-specific (different control paper_id per distance)
    """

    t = treated.astype("int64")
    if not t.isin([0, 1]).all():
        raise ValueError("treated must be 0/1 to build paper_id")

    # Prefer compact int64 IDs when match_id is safely numeric.
    try:
        m = match_id.astype("int64")
        d = distance.astype("int64")
        if (d < 0).any():
            raise ValueError("distance must be non-negative")
        # Reserve codes:
        #   treated: 1
        #   control at distance d: 2 + d  (so d=1..6 -> 3..8)
        code = pd.Series(np.where(t.to_numpy() == 1, 1, 2 + d.to_numpy()), index=match_id.index).astype("int64")
        M = 32  # small multiplier; must exceed max code
        max_safe = (np.iinfo(np.int64).max - int(code.max())) // M
        if m.max() > max_safe or m.min() < 0:
            raise OverflowError("match_id out of safe range for int64 paper_id")
        return (m * M + code).astype("int64")
    except Exception:
        # String fallback: stable treated IDs; distance-specific controls.
        m_str = match_id.astype(str)
        d_str = distance.astype(str)
        out = m_str + "_t" + t.astype(str)
        is_control = (t == 0)
        out.loc[is_control] = m_str.loc[is_control] + "_c_d" + d_str.loc[is_control]
        return out

def _build_panel_vector_current(
    raw: pd.DataFrame,
    outcome_col: str,
    years: int,
    vector_offset: int,
    distance_value: int = 1,
) -> pd.DataFrame:
    required = ["corpusid", "treated", "publicationdate", "RetractionDate", outcome_col]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Vector format missing columns: {missing}")

    df = raw[required].copy()
    df = df.rename(columns={"corpusid": "match_id", "RetractionDate": "retraction_date"})
    df["treated"] = df["treated"].astype(int)
    _ensure_1to1_matches(df, "match_id")

    df["pub_year"] = _to_year(df["publicationdate"])
    df["retract_year"] = _to_year(df["retraction_date"])
    if "distance" in raw.columns:
        df["distance"] = raw.loc[df.index, "distance"].astype(int)
    else:
        df["distance"] = int(distance_value)
    # paper_id: treated paper stable across distances; control paper distance-specific (controls can differ by distance).
    df["paper_id"] = _paper_id_vector_current(df["match_id"], df["treated"], df["distance"])

    vec_lens = df[outcome_col].apply(lambda x: len(x) if x is not None else np.nan)
    if vec_lens.isna().any():
        raise ValueError(f"Found null vectors in '{outcome_col}'.")
    min_len = int(vec_lens.min())
    if min_len < vector_offset + years:
        raise ValueError(
            f"Vector '{outcome_col}' too short for years={years}, vector_offset={vector_offset}. "
            f"min length={min_len}"
        )

    vec_mat = np.vstack(df[outcome_col].to_numpy())
    y_mat = vec_mat[:, vector_offset : vector_offset + years]
    y = y_mat.reshape(-1).astype(float)

    n = len(df)
    ages = np.tile(np.arange(1, years + 1, dtype=int), n)
    panel = pd.DataFrame(
        {
            "match_id": np.repeat(df["match_id"].to_numpy(), years),
            "paper_id": np.repeat(df["paper_id"].to_numpy(), years),
            "treated": np.repeat(df["treated"].to_numpy(), years),
            "distance": np.repeat(df["distance"].to_numpy(), years),
            "age": ages,
            "pub_year": np.repeat(df["pub_year"].to_numpy(), years),
            "retract_year": np.repeat(df["retract_year"].to_numpy(), years),
        }
    )
    panel["year"] = panel["pub_year"] + panel["age"] - 1
    panel["post_ret"] = (panel["year"] >= panel["retract_year"]).astype(int)
    panel["y"] = y
    return panel


def _build_panel_wide_paper(raw: pd.DataFrame) -> pd.DataFrame:
    required = ["match_id", "treated", "pub_year", "distance", "retract_year"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Wide paper format missing columns: {missing}")

    df = raw.copy()
    if "paper_id" not in df.columns:
        df["paper_id"] = df["match_id"].astype(str) + "_" + df["treated"].astype(str)

    df = _inherit_within_match(df, ["distance", "retract_year"], match_col="match_id")
    df["treated"] = df["treated"].astype(int)
    _ensure_1to1_matches(df.drop_duplicates(["match_id", "treated"]), "match_id")

    # Respect --years for wide format to control memory footprint.
    # (Wide format has at most cites_y1..cites_y10.)
    years = 10
    # If caller passed fewer years by dropping columns before calling, handle that too.
    for k in range(10, 0, -1):
        if f"cites_y{k}" in df.columns:
            years = k
            break
    y_cols = [f"cites_y{i}" for i in range(1, years + 1)]
    missing_y = [c for c in y_cols if c not in df.columns]
    if missing_y:
        raise ValueError(f"Wide paper format missing yearly columns: {missing_y}")

    long_df = df.melt(
        id_vars=[c for c in df.columns if c not in y_cols],
        value_vars=y_cols,
        var_name="age_col",
        value_name="y",
    )
    long_df["age"] = long_df["age_col"].str.extract(r"(\d+)$").astype(int)
    long_df = long_df.drop(columns=["age_col"])

    long_df["pub_year"] = long_df["pub_year"].astype(int)
    long_df["retract_year"] = long_df["retract_year"].astype(int)
    long_df["distance"] = long_df["distance"].astype(int)
    long_df["year"] = long_df["pub_year"] + long_df["age"] - 1
    long_df["post_ret"] = (long_df["year"] >= long_df["retract_year"]).astype(int)
    long_df["y"] = long_df["y"].fillna(0).astype(float)
    return long_df[["match_id", "paper_id", "treated", "distance", "age", "year", "pub_year", "retract_year", "post_ret", "y"]]


def _build_panel_long(raw: pd.DataFrame, outcome_col: str) -> pd.DataFrame:
    df = raw.copy()
    if "paper_id" not in df.columns:
                raise ValueError(
                    "--unified-5b is not supported in the legacy streaming engine (--time-fe age) "
                    "because it would rely on an explicitly approximate cross-distance test. "
                    "Use --time-fe year (matched 1:1 streaming engine) with --unified-5b instead."
                )

    if "distance" not in df.columns:
        df["distance"] = 1

    if outcome_col not in df.columns:
        if "y" in df.columns:
            outcome_col = "y"
        else:
            raise ValueError(f"Long format missing outcome column '{outcome_col}'")

    df["treated"] = df["treated"].astype(int)
    df["distance"] = df["distance"].astype(int)
    df["age"] = df["age"].astype(int)
    _ensure_1to1_matches(df.drop_duplicates(["match_id", "treated"]), "match_id")

    df = _inherit_within_match(df, ["retract_year", "pub_year"], match_col="match_id")

    if "year" not in df.columns:
        if "pub_year" in df.columns:
            df["year"] = df["pub_year"].astype(int) + df["age"] - 1
        else:
            raise ValueError("Long format needs 'year' or 'pub_year' to compute year")

    if "post_ret" not in df.columns:
        if "retract_year" not in df.columns:
            raise ValueError("Long format needs 'post_ret' or 'retract_year' to compute it")
        df["retract_year"] = df["retract_year"].astype(int)
        df["post_ret"] = (df["year"].astype(int) >= df["retract_year"]).astype(int)

    df["y"] = df[outcome_col].astype(float)
    keep = ["match_id", "paper_id", "treated", "distance", "age", "year", "pub_year", "retract_year", "post_ret", "y"]
    for c in keep:
        if c not in df.columns:
            df[c] = np.nan
    return df[keep]


def load_panel(
    path: str,
    outcome_col: str,
    years: int,
    vector_offset: int,
    transform: str,
    forced_distance: int | None = None,
) -> tuple[pd.DataFrame, str, str]:
    # Read only the necessary columns to reduce memory.
    colnames = _parquet_column_names(path)
    cols_set = set(colnames)
    read_cols: list[str] | None = None

    # Decide likely format from metadata (no full read).
    if colnames:
        if {"corpusid", "treated", "publicationdate", "RetractionDate", outcome_col}.issubset(cols_set):
            read_cols = ["corpusid", "treated", "publicationdate", "RetractionDate", outcome_col]
            if "distance" in cols_set:
                read_cols.append("distance")
        elif any(re.match(r"^cites_y\d+$", str(c)) for c in colnames) and {"match_id", "treated", "pub_year", "distance"}.issubset(cols_set):
            # wide format: read only years requested
            y_cols = [f"cites_y{i}" for i in range(1, min(years, 10) + 1)]
            read_cols = ["match_id", "treated", "pub_year", "distance", "retract_year"] + y_cols
            if "paper_id" in cols_set:
                read_cols.append("paper_id")
        elif {"match_id", "treated", "age"}.issubset(cols_set):
            read_cols = ["match_id", "treated", "age"]
            for c in ["paper_id", "distance", "year", "pub_year", "retract_year", "post_ret", outcome_col, "y"]:
                if c in cols_set:
                    read_cols.append(c)

    raw = pd.read_parquet(path, columns=read_cols)
    fmt = _detect_format(raw, outcome_col)
    if fmt == "vector_current":
        dval = int(forced_distance) if forced_distance is not None else 1
        panel = _build_panel_vector_current(
            raw,
            outcome_col=outcome_col,
            years=years,
            vector_offset=vector_offset,
            distance_value=dval,
        )
        outcome_label = outcome_col
    elif fmt == "wide_paper":
        panel = _build_panel_wide_paper(raw)
        outcome_label = "cites"
    elif fmt == "long_panel":
        panel = _build_panel_long(raw, outcome_col=outcome_col)
        outcome_label = outcome_col
    else:
        raise ValueError(
            "Unrecognized input format. Present columns: "
            f"{list(raw.columns)}\n"
            "Supported: vector_current (corpusid/publicationdate/RetractionDate/vector), wide_paper (cites_y1..10), long_panel."
        )

    if transform == "log1p":
        panel["y"] = np.log1p(panel["y"].astype(float))
        outcome_label = f"log1p({outcome_label})"
    elif transform != "none":
        raise ValueError("--transform must be 'none' or 'log1p'")

    if forced_distance is not None:
        if "distance" not in panel.columns:
            panel["distance"] = int(forced_distance)
        else:
            uniq = sorted(panel["distance"].dropna().astype(int).unique().tolist())
            if len(uniq) > 1:
                raise ValueError(
                    f"File '{path}' contains multiple distance values {uniq}; cannot force distance={forced_distance}."
                )
            if len(uniq) == 1 and int(uniq[0]) != int(forced_distance):
                raise ValueError(
                    f"File '{path}' has distance={uniq[0]} but filename/forced_distance={forced_distance}."
                )
        panel["distance"] = int(forced_distance)

    return panel, outcome_label, fmt


def load_panels(
    paths: list[str],
    outcome_col: str,
    years: int,
    vector_offset: int,
    transform: str,
) -> tuple[pd.DataFrame, str]:
    if not paths:
        raise ValueError("No data files provided")

    panels: list[pd.DataFrame] = []
    outcome_label: str | None = None
    for p in paths:
        forced_d = _infer_distance_from_path(p)
        panel_i, label_i, fmt_i = load_panel(
            p,
            outcome_col=outcome_col,
            years=years,
            vector_offset=vector_offset,
            transform=transform,
            forced_distance=forced_d,
        )
        if outcome_label is None:
            outcome_label = label_i
        elif outcome_label != label_i:
            # Keep the first label; mixed labels usually means mixed formats.
            pass

        if forced_d is not None and fmt_i != "vector_current":
            # For non-vector_current formats we keep the old safety behavior: namespace IDs across
            # per-distance files to avoid accidental collisions when stacking.
            panel_i = _namespace_ids(panel_i, distance=forced_d, source=p)
        else:
            # IMPORTANT: for vector_current, do NOT namespace match_id (== corpusid treated-paper id).
            # corpusid can repeat across distances; we want pooled/unified clustering to account for that.
            panel_i["source_file"] = p
        panels.append(panel_i)

    combined = pd.concat(panels, axis=0, ignore_index=True)
    return combined, (outcome_label or outcome_col)


def run_tasks_one_distance(
    panel: pd.DataFrame,
    outcome_label: str,
    distance_value: int,
    pretrend_leads: int,
    time_fe: str,
    *,
    estimator: str = "ols",
    event_study: bool = False,
    es_leads: int = 3,
    es_lags: int = 5,
) -> None:
    """Run Tasks 1–5A for a single distance (memory-friendly)."""
    panel = panel.copy()
    panel["treated"] = panel["treated"].astype(int)
    panel["distance"] = panel["distance"].astype(int)
    panel["post_ret"] = panel["post_ret"].astype(int)
    sub = panel[panel["distance"] == int(distance_value)]
    if len(sub) == 0:
        raise ValueError(f"No observations for distance={distance_value}")

    print(f"\n========== Distance = {distance_value} ==========")
    time_cols = _time_fe_cols(time_fe)

    res1 = fit_fe(sub, x_cols=["treated"], fe_cols=["match_id"] + time_cols, cluster_col="match_id", estimator=estimator)
    _print_result(f"Task 1: treated vs control ({outcome_label})", res1)

    pre = sub[sub["post_ret"] == 0]
    if len(pre) > 0:
        res2 = fit_fe(pre, x_cols=["treated"], fe_cols=["match_id"] + time_cols, cluster_col="match_id", estimator=estimator)
        _print_result(f"Task 2: pre-retraction treated vs control ({outcome_label})", res2)
    else:
        print("Task 2 skipped (no pre-retraction observations)")

    post = sub[sub["post_ret"] == 1]
    if len(post) > 0:
        res3 = fit_fe(post, x_cols=["treated"], fe_cols=["match_id"] + time_cols, cluster_col="match_id", estimator=estimator)
        _print_result(f"Task 3: post-retraction treated vs control ({outcome_label})", res3)
    else:
        print("Task 3 skipped (no post-retraction observations)")

    treated_only = sub[sub["treated"] == 1]
    if treated_only["post_ret"].nunique() >= 2:
        res4 = fit_fe(treated_only, x_cols=["post_ret"], fe_cols=["paper_id"] + time_cols, cluster_col="match_id", estimator=estimator)
        _print_result(f"Task 4: treated-only post vs pre ({outcome_label})", res4)
    else:
        print("Task 4 skipped (treated-only has no pre/post variation)")

    did = sub.copy()
    did["treated_post"] = did["treated"] * did["post_ret"]
    res5a = fit_fe(
        did,
        x_cols=["post_ret", "treated_post"],
        fe_cols=["paper_id"] + time_cols,
        cluster_col="match_id",
        estimator=estimator,
    )
    msg, _ = pretrend_leads_test(did, max_lead=pretrend_leads, time_fe=time_fe)
    _print_result(
        f"Task 5A (DiD): distance={distance_value} ({outcome_label})",
        res5a,
        effect_note=f"  DiD effect is coef on treated_post. {msg}",
    )

    if event_study:
        try:
            tab, msg2 = event_study_fe(did, leads=int(es_leads), lags=int(es_lags), time_fe=time_fe, estimator=estimator)
            rows = [
                (int(r.tau), float(r.est), float(r.se), float(r.t), float(r.p), float(r.ci_low), float(r.ci_high), int(r.n_clusters))
                for r in tab.itertuples(index=False)
            ]
            _print_event_study_table(
                f"Event-study (TWFE): distance={distance_value} (ref tau=-1). {msg2}",
                rows,
            )
        except Exception as e:
            print(f"Event-study skipped for distance={distance_value}: {e}")


def _group_means(values: np.ndarray, group: np.ndarray, n_groups: int) -> np.ndarray:
    sums = np.bincount(group, weights=values, minlength=n_groups)
    counts = np.bincount(group, minlength=n_groups)
    means = sums / counts
    return means[group]


def _two_way_demean(
    y: np.ndarray,
    X: np.ndarray,
    g1: np.ndarray,
    n_g1: int,
    g2: np.ndarray,
    n_g2: int,
    n_iter: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """Iterative demeaning for two-way fixed effects (g1 and g2)."""
    y_res = y.astype(float).copy()
    X_res = X.astype(float).copy()

    for _ in range(n_iter):
        y_res -= _group_means(y_res, g1, n_g1)
        y_res -= _group_means(y_res, g2, n_g2)

        for j in range(X_res.shape[1]):
            col = X_res[:, j]
            col -= _group_means(col, g1, n_g1)
            col -= _group_means(col, g2, n_g2)
            X_res[:, j] = col

    return y_res, X_res


def _k_way_demean(
    y: np.ndarray,
    X: np.ndarray,
    groups: list[np.ndarray],
    n_groups: list[int],
    n_iter: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Iterative demeaning for k-way fixed effects using alternating projections."""
    y_res = y.astype(float).copy()
    X_res = X.astype(float).copy()

    for _ in range(n_iter):
        for g, ng in zip(groups, n_groups):
            y_res -= _group_means(y_res, g, ng)
            for j in range(X_res.shape[1]):
                col = X_res[:, j]
                col -= _group_means(col, g, ng)
                X_res[:, j] = col
    return y_res, X_res


@dataclass
class OLSResult:
    coef_names: list[str]
    beta: np.ndarray
    se: np.ndarray
    t: np.ndarray
    p: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    vcov: np.ndarray
    nobs: int
    n_clusters: int


def ols_cluster_robust(
    y: np.ndarray,
    X: np.ndarray,
    cluster: np.ndarray,
    coef_names: list[str],
) -> OLSResult:
    """OLS with cluster-robust SE (Arellano-style) using numpy only."""
    nobs, k = X.shape
    if nobs <= k:
        raise ValueError(f"Not enough observations (nobs={nobs}) for k={k} regressors")

    XtX = X.T @ X
    Xty = X.T @ y
    try:
        beta = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(XtX, Xty, rcond=None)[0]
    u = y - (X @ beta)

    # Cluster scores: S_g = sum_{i in g} x_i * u_i
    cluster_codes, _ = pd.factorize(cluster, sort=False)
    n_clusters = int(cluster_codes.max()) + 1
    if n_clusters < 5:
        raise ValueError(f"Too few clusters for cluster-robust SE: {n_clusters}")

    scores = np.zeros((n_clusters, k), dtype=float)
    np.add.at(scores, cluster_codes, X * u[:, None])
    meat = scores.T @ scores

    try:
        bread = np.linalg.solve(XtX, np.eye(k, dtype=float))
    except np.linalg.LinAlgError:
        bread = np.linalg.pinv(XtX)
    V = bread @ meat @ bread
    # CR1 small-sample correction (standard; effect is tiny with large clusters)
    G = n_clusters
    scale = (G / (G - 1)) * ((nobs - 1) / (nobs - X.shape[1]))
    V = V * scale

    diagV = np.diag(V).astype(float)
    bad_var = (~np.isfinite(diagV)) | (diagV <= 0.0)
    if np.any(bad_var):
        bad_names = [coef_names[i] for i in np.where(bad_var)[0].tolist() if i < len(coef_names)]
        print(
            "[note] OLS: non-positive or non-finite variance for coefficients "
            f"{bad_names}. SE/t/p are set to NA for these terms; results may be numerically unstable."
        )

    se = np.sqrt(np.where(bad_var, np.nan, diagV))
    t_stat = np.divide(beta, se, out=np.full_like(beta, np.nan, dtype=float), where=np.isfinite(se) & (se > 0))

    stats = _require_scipy_stats()
    df = n_clusters - 1
    p = 2.0 * stats.t.sf(np.abs(t_stat), df=df)
    p = np.clip(p, 0.0, 1.0)
    t_crit = float(stats.t.ppf(0.975, df=df))

    ci_low = beta - t_crit * se
    ci_high = beta + t_crit * se

    beta = np.asarray(beta, dtype=float).reshape(-1)
    se = np.asarray(se, dtype=float).reshape(-1)
    t_stat = np.asarray(t_stat, dtype=float).reshape(-1)
    p = np.asarray(p, dtype=float).reshape(-1)
    ci_low = np.asarray(ci_low, dtype=float).reshape(-1)
    ci_high = np.asarray(ci_high, dtype=float).reshape(-1)

    return OLSResult(
        coef_names=list(coef_names),
        beta=beta,
        se=se,
        t=t_stat,
        p=p,
        ci_low=ci_low,
        ci_high=ci_high,
        vcov=np.asarray(V, dtype=float),
        nobs=int(nobs),
        n_clusters=int(n_clusters),
    )


def _weighted_group_means(values: np.ndarray, group: np.ndarray, n_groups: int, weights: np.ndarray) -> np.ndarray:
    wsum = np.bincount(group, weights=weights, minlength=n_groups)
    wy = np.bincount(group, weights=weights * values, minlength=n_groups)
    mean = wy / np.maximum(wsum, 1e-30)
    return mean[group]


def _weighted_k_way_demean_vector(
    y: np.ndarray,
    groups: list[np.ndarray],
    n_groups: list[int],
    weights: np.ndarray,
    n_iter: int = 20,
) -> np.ndarray:
    y_res = y.astype(float).copy()
    w = weights.astype(float)
    for _ in range(n_iter):
        for g, ng in zip(groups, n_groups):
            y_res -= _weighted_group_means(y_res, g, ng, w)
    return y_res


def _weighted_k_way_demean(
    y: np.ndarray,
    X: np.ndarray,
    groups: list[np.ndarray],
    n_groups: list[int],
    weights: np.ndarray,
    n_iter: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    y_res = y.astype(float).copy()
    X_res = X.astype(float).copy()
    w = weights.astype(float)
    for _ in range(n_iter):
        for g, ng in zip(groups, n_groups):
            y_res -= _weighted_group_means(y_res, g, ng, w)
            for j in range(X_res.shape[1]):
                col = X_res[:, j]
                col -= _weighted_group_means(col, g, ng, w)
                X_res[:, j] = col
    return y_res, X_res


def _ppml_wls_absorb_fe(
    z: np.ndarray,
    X: np.ndarray,
    groups: list[np.ndarray],
    n_groups: list[int],
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve one IRLS WLS step for PPML with additive fixed effects absorbed.

    Returns (beta, eta) where eta = X beta + FE_fitted.
    """

    if len(groups) == 0:
        X_res, z_res = X, z
    else:
        z_res, X_res = _weighted_k_way_demean(z, X, groups, n_groups, weights, n_iter=20)

    W = weights.reshape(-1, 1)
    XtWX = X_res.T @ (W * X_res)
    XtWz = X_res.T @ (weights * z_res)
    try:
        beta = np.linalg.solve(XtWX, XtWz)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(XtWX, XtWz, rcond=None)[0]

    # Recover eta = X beta + FE_fitted using FE projection on r = z - Xbeta
    r = z - (X @ beta)
    if len(groups) == 0:
        eta = X @ beta
    else:
        r_res = _weighted_k_way_demean_vector(r, groups, n_groups, weights, n_iter=20)
        fe_fit = r - r_res
        eta = (X @ beta) + fe_fit
    return beta.reshape(-1), eta.reshape(-1)


def ppml_fe_cluster_robust(
    y: np.ndarray,
    X: np.ndarray,
    groups: list[np.ndarray],
    n_groups: list[int],
    cluster: np.ndarray,
    coef_names: list[str],
    *,
    max_iter: int = 100,
    tol: float = 1e-9,
) -> OLSResult:
    """PPML (Poisson pseudo-ML) with absorbed FE via IRLS and cluster-robust SE.

    Intended as a robustness estimator for non-stream runs. For huge data, prefer streaming OLS.
    """

    if (y < 0).any():
        raise ValueError("PPML requires nonnegative outcomes y")

    nobs, k = X.shape
    if nobs <= k:
        raise ValueError(f"Not enough observations (nobs={nobs}) for k={k} regressors")

    y = y.astype(float)
    X = X.astype(float)

    # Initialize eta at log(mean(y))
    y_mean = float(np.mean(y))
    eta = np.full(nobs, math.log(max(y_mean, 1e-8)), dtype=float)
    beta = np.zeros(k, dtype=float)

    # Numerical guardrails: bound eta to stabilize exp() and prevent extreme weights.
    eta_clip_min = -30.0
    eta_clip_max = 30.0
    clip_share_last = 0.0
    converged = False

    for it in range(int(max_iter)):
        # Clip eta to avoid overflow in exp
        eta_clip = np.clip(eta, eta_clip_min, eta_clip_max)
        clip_share_last = float(np.mean((eta <= eta_clip_min) | (eta >= eta_clip_max)))
        mu = np.exp(eta_clip)
        mu = np.maximum(mu, 1e-12)

        w = mu
        z = eta_clip + (y - mu) / mu

        beta_new, eta_new = _ppml_wls_absorb_fe(z, X, groups, n_groups, w)
        # Keep eta itself bounded; FE recovery can otherwise create huge eta values
        # even if exp() is clipped later, harming numerical stability.
        eta_new = np.clip(eta_new, eta_clip_min, eta_clip_max)

        delta = float(np.max(np.abs(beta_new - beta)))
        beta = beta_new
        eta = eta_new
        if delta < tol:
            converged = True
            break

    if not converged:
        raise RuntimeError(
            "PPML did not converge within max_iter; aborting to avoid unreliable inference. "
            "Try increasing --ppml-max-iter, loosening --ppml-tol, or using OLS/log1p."
        )

    if clip_share_last >= 0.01:
        print(
            f"[note] PPML: {clip_share_last:.2%} of eta values hit the clip boundary "
            f"[{eta_clip_min}, {eta_clip_max}]. Results/SE may be numerically unstable."
        )

    eta_clip = np.clip(eta, eta_clip_min, eta_clip_max)
    mu = np.exp(eta_clip)
    mu = np.maximum(mu, 1e-12)

    # Compute bread using final weights and residualized X
    if len(groups) == 0:
        X_res = X
    else:
        _, X_res = _weighted_k_way_demean(np.zeros_like(y), X, groups, n_groups, mu, n_iter=20)
    XtWX = X_res.T @ (mu.reshape(-1, 1) * X_res)
    try:
        bread = np.linalg.solve(XtWX, np.eye(k, dtype=float))
    except np.linalg.LinAlgError:
        bread = np.linalg.pinv(XtWX)

    # Cluster meat using score = x_res * (y - mu)
    resid = (y - mu).astype(float)
    cluster_codes, _ = pd.factorize(cluster, sort=False)
    G = int(cluster_codes.max()) + 1
    if G < 5:
        raise ValueError(f"Too few clusters for cluster-robust SE: {G}")
    scores = np.zeros((G, k), dtype=float)
    np.add.at(scores, cluster_codes, X_res * resid[:, None])
    meat = scores.T @ scores
    V = bread @ meat @ bread

    # CR1 correction
    scale = (G / (G - 1)) * ((nobs - 1) / (nobs - k))
    V = V * scale

    diagV = np.diag(V).astype(float)
    bad_var = (~np.isfinite(diagV)) | (diagV <= 0.0)
    if np.any(bad_var):
        bad_names = [coef_names[i] for i in np.where(bad_var)[0].tolist() if i < len(coef_names)]
        print(
            "[note] PPML: non-positive or non-finite variance for coefficients "
            f"{bad_names}. SE/t/p are set to NA for these terms; results may be numerically unstable."
        )

    se = np.sqrt(np.where(bad_var, np.nan, diagV))
    t_stat = np.divide(beta, se, out=np.full_like(beta, np.nan, dtype=float), where=np.isfinite(se) & (se > 0))
    stats = _require_scipy_stats()
    df = G - 1
    p = 2.0 * stats.t.sf(np.abs(t_stat), df=df)
    p = np.clip(p, 0.0, 1.0)
    t_crit = float(stats.t.ppf(0.975, df=df))

    ci_low = beta - t_crit * se
    ci_high = beta + t_crit * se

    return OLSResult(
        coef_names=list(coef_names),
        beta=np.asarray(beta, dtype=float).reshape(-1),
        se=np.asarray(se, dtype=float).reshape(-1),
        t=np.asarray(t_stat, dtype=float).reshape(-1),
        p=np.asarray(p, dtype=float).reshape(-1),
        ci_low=np.asarray(ci_low, dtype=float).reshape(-1),
        ci_high=np.asarray(ci_high, dtype=float).reshape(-1),
        vcov=np.asarray(V, dtype=float),
        nobs=int(nobs),
        n_clusters=int(G),
    )


def _print_event_study_table(
    title: str,
    rows: list[tuple[int, float, float, float, float, float, float, int]],
) -> None:
    """Each row: (tau, est, se, t, p, ci_low, ci_high, n_clusters_used)."""
    print(f"\n{title}")
    print("  tau           est        se        t        p                 95% CI        clusters")
    for tau, est, se, t, p, lo, hi, g in rows:
        p_txt = _format_p(float(p))
        se_f = float(se)
        if not np.isfinite(se_f):
            se_txt = "NA"
        else:
            se_txt = f"{se_f: .6f}" if abs(se_f) >= 1e-6 else f"{se_f: .6e}"

        est_txt = _fmt_float(float(est), " .6f")
        t_txt = _fmt_float(float(t), " .3f")
        lo_txt = _fmt_float(float(lo), " .6f")
        hi_txt = _fmt_float(float(hi), " .6f")
        print(f"  {int(tau):<6} {est_txt}  {se_txt}  {t_txt}  {p_txt:<12}  [{lo_txt}, {hi_txt}]   {int(g):,}")


def event_study_fe(
    panel: pd.DataFrame,
    *,
    leads: int,
    lags: int,
    time_fe: str,
    estimator: str,
) -> tuple[pd.DataFrame, str]:
    """Run TWFE event-study: y ~ sum_k 1[tau==k]*treated + FE(paper) + FE(time).

    Reference period is tau=-1 (omitted).
    Returns (table_df, pretrend_msg).
    """

    if leads < 0 or lags < 0:
        raise ValueError("event-study leads/lags must be >= 0")
    if panel[["year", "retract_year"]].isna().any().any():
        raise ValueError("event-study requires non-missing year and retract_year")

    # Event-study is defined on calendar year; require year to be part of time FE.
    t = time_fe.strip().lower()
    if t not in {"year", "year+age", "age+year"}:
        raise ValueError("event-study requires --time-fe year (or year+age)")

    tau = panel["year"].astype(int) - panel["retract_year"].astype(int)
    did = panel.copy()

    tau_vals: list[int] = []
    for k in range(-int(leads), int(lags) + 1):
        if k == -1:
            continue
        tau_vals.append(int(k))

    x_cols: list[str] = []
    for k in tau_vals:
        name = f"tau_{k:+d}".replace("+", "p").replace("-", "m")
        did[name] = (did["treated"] * (tau == k)).astype(int)
        if did[name].sum() > 0:
            x_cols.append(name)

    if not x_cols:
        return pd.DataFrame(columns=["tau", "est", "se", "t", "p", "ci_low", "ci_high", "n_clusters"]), "Pretrend test skipped (no event-study observations)"

    fe_cols = ["paper_id"] + _time_fe_cols(time_fe)
    res = fit_fe(did, x_cols=x_cols, fe_cols=fe_cols, cluster_col="match_id", estimator=estimator)

    # Table
    rows = []
    for k in tau_vals:
        name = f"tau_{k:+d}".replace("+", "p").replace("-", "m")
        if name not in res.coef_names:
            continue
        i = _coef_index(res, name)
        rows.append(
            {
                "tau": int(k),
                "est": float(res.beta[i]),
                "se": float(res.se[i]),
                "t": float(res.t[i]),
                "p": float(res.p[i]),
                "ci_low": float(res.ci_low[i]),
                "ci_high": float(res.ci_high[i]),
                "n_clusters": int(res.n_clusters),
            }
        )
    tab = pd.DataFrame(rows).sort_values("tau")

    # Pretrend (leads) joint Wald
    lead_cols: list[str] = []
    for k in range(2, int(leads) + 1):
        name = f"tau_{-k:+d}".replace("+", "p").replace("-", "m")
        if name in res.coef_names:
            lead_cols.append(name)
    if lead_cols:
        stat, df, p = wald_test(res, lead_cols)
        p0, used_approx = _chi2_p_value(stat, df)
        msg = f"Pretrend leads Wald: chi2({df})={stat:.3f}, p={_format_chi2_p(p0, used_approx)}"
    else:
        msg = "Pretrend test skipped (no lead observations)"

    return tab, msg


def _format_p(p: float) -> str:
    p = _clip_p01(float(p))
    if not np.isfinite(p):
        return "NA"
    if p == 0.0:
        return "<1e-300"
    return f"{p:.3e}"


def _fmt_float(x: float, spec: str) -> str:
    """Format float with spec; return 'NA' if not finite."""
    xx = float(x)
    if not np.isfinite(xx):
        return "NA"
    return format(xx, spec)


def _clip_p01(p: float) -> float:
    """Clip a (finite) p-value into [0,1] to avoid numerical overshoots."""
    if not np.isfinite(p):
        return float("nan")
    if p < 0.0:
        return 0.0
    if p > 1.0:
        return 1.0
    return float(p)


def _vector_list_col_to_2d_fixed(list_array, *, offset: int, years: int) -> np.ndarray:
    """Convert Arrow large_list column with fixed list length to a dense 2D NumPy array.

    This is zero-copy-ish and avoids Python list materialization.
    """
    import numpy as _np

    # pyarrow may return a ChunkedArray when reading row groups; normalize to a single array.
    if hasattr(list_array, "combine_chunks"):
        try:
            list_array = list_array.combine_chunks()
        except Exception:
            pass

    off = _np.asarray(list_array.offsets)
    if len(off) < 2:
        return _np.zeros((0, years), dtype=float)
    L = int(off[1] - off[0])
    if not _np.all(off == _np.arange(len(off)) * L):
        raise ValueError("Outcome list column is not fixed-length per row; cannot stream efficiently")
    vals = _np.asarray(list_array.values)
    mat = vals.reshape(len(off) - 1, L)
    if L < offset + years:
        raise ValueError(f"Outcome vector too short: length={L}, need offset={offset} + years={years}")
    return mat[:, offset : offset + years].astype(float, copy=False)


def _cluster_robust_1reg(
    x: np.ndarray,
    y: np.ndarray,
    cluster_scores: np.ndarray,
    nobs: int,
    n_clusters: int,
    coef_name: str,
) -> OLSResult:
    """Cluster-robust OLS for single regressor (no intercept) given x,y already residualized.

    cluster_scores should be s_g = sum_{i in g} x_i * u_i, but since u depends on beta,
    this helper is only used after beta is known in streaming calculations.
    """
    XtX = float(np.dot(x, x))
    if XtX == 0.0:
        raise ValueError(f"Regressor '{coef_name}' has zero variance after FE demeaning")
    beta = float(np.dot(x, y) / XtX)

    meat = float(np.dot(cluster_scores, cluster_scores))
    bread = 1.0 / XtX
    V = bread * meat * bread

    # CR1 correction
    G = n_clusters
    k = 1
    scale = (G / (G - 1)) * ((nobs - 1) / (nobs - k))
    V = V * scale
    se = math.sqrt(V) if V >= 0 else float("nan")
    t_stat = beta / se if se else float("nan")

    stats = _require_scipy_stats()
    df = G - 1
    if np.isfinite(t_stat):
        p = float(2.0 * stats.t.sf(abs(t_stat), df=df))
        p = _clip_p01(p)
        tcrit = float(stats.t.ppf(0.975, df=df))
    else:
        p = float("nan")
        tcrit = float("nan")

    ci_low = beta - tcrit * se
    ci_high = beta + tcrit * se
    return OLSResult(
        coef_names=[coef_name],
        beta=np.array([beta], dtype=float),
        se=np.array([se], dtype=float),
        t=np.array([t_stat], dtype=float),
        p=np.array([p], dtype=float),
        ci_low=np.array([ci_low], dtype=float),
        ci_high=np.array([ci_high], dtype=float),
        vcov=np.array([[V]], dtype=float),
        nobs=int(nobs),
        n_clusters=int(n_clusters),
    )


def _cluster_robust_kreg(
    XtX: np.ndarray,
    Xty: np.ndarray,
    meat: np.ndarray,
    nobs: int,
    n_clusters: int,
    coef_names: list[str],
) -> OLSResult:
    k = int(XtX.shape[0])
    try:
        beta = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(XtX, Xty, rcond=None)[0]
    try:
        bread = np.linalg.solve(XtX, np.eye(k, dtype=float))
    except np.linalg.LinAlgError:
        bread = np.linalg.pinv(XtX)
    V = bread @ meat @ bread

    G = n_clusters
    scale = (G / (G - 1)) * ((nobs - 1) / (nobs - k))
    V = V * scale
    diagV = np.diag(V).astype(float)
    bad_var = (~np.isfinite(diagV)) | (diagV <= 0.0)
    if np.any(bad_var):
        bad_names = [coef_names[i] for i in np.where(bad_var)[0].tolist() if i < len(coef_names)]
        print(
            "[note] OLS(stream): non-positive or non-finite variance for coefficients "
            f"{bad_names}. SE/t/p are set to NA for these terms; results may be numerically unstable."
        )

    se = np.sqrt(np.where(bad_var, np.nan, diagV))
    t_stat = np.divide(beta, se, out=np.full_like(beta, np.nan, dtype=float), where=np.isfinite(se) & (se > 0))

    stats = _require_scipy_stats()
    df = G - 1
    p = 2.0 * stats.t.sf(np.abs(t_stat), df=df)  # Calculate p-value
    p = np.clip(p, 0.0, 1.0)
    tcrit = float(stats.t.ppf(0.975, df=df))  # Critical t-value for 95% CI

    ci_low = beta - tcrit * se
    ci_high = beta + tcrit * se
    return OLSResult(
        coef_names=list(coef_names),
        beta=np.asarray(beta, dtype=float).reshape(-1),
        se=np.asarray(se, dtype=float).reshape(-1),
        t=np.asarray(t_stat, dtype=float).reshape(-1),
        p=np.asarray(p, dtype=float).reshape(-1),
        ci_low=np.asarray(ci_low, dtype=float).reshape(-1),
        ci_high=np.asarray(ci_high, dtype=float).reshape(-1),
        vcov=np.asarray(V, dtype=float),
        nobs=int(nobs),
        n_clusters=int(n_clusters),
    )


def run_stream_vector_tasks(
    path: str,
    *,
    outcome_col: str,
    years: int,
    vector_offset: int,
    transform: str,
    distance_value: int,
    pretrend_leads: int,
) -> OLSResult:
    """Low-memory runner for very large vector_current parquet.

    This avoids expanding to a long panel DataFrame. It supports ONLY age fixed effects.
    Cluster-robust SE are computed at match (pair) level using corpusid (which is match_id in vector_current).
    """
    import pyarrow.parquet as pq  # type: ignore
    import pyarrow.compute as pc  # type: ignore

    if years < 1:
        raise ValueError("--years must be >= 1")
    if transform not in {"none", "log1p"}:
        raise ValueError("--transform must be 'none' or 'log1p'")
    if pretrend_leads not in (0, 1) and pretrend_leads != 0:
        # event-study leads needs calendar year tau; streaming engine does not implement it.
        pass

    cols = ["corpusid", "treated", "publicationdate", "RetractionDate", outcome_col]
    pf = pq.ParquetFile(path)
    schema_cols = set(pf.schema_arrow.names)
    missing = [c for c in cols if c not in schema_cols]
    if missing:
        raise ValueError(f"Vector format missing columns {missing} in file {path}")

    T = int(years)
    ages = np.arange(1, T + 1, dtype=np.int16)

    # ---------- Pass 0: compute age means for each task's analytic sample ----------
    sum_y_all = np.zeros(T, dtype=np.float64)
    cnt_all = np.zeros(T, dtype=np.int64)
    sum_t_all = np.zeros(T, dtype=np.float64)

    sum_y_pre = np.zeros(T, dtype=np.float64)
    cnt_pre = np.zeros(T, dtype=np.int64)
    sum_t_pre = np.zeros(T, dtype=np.float64)

    sum_y_post = np.zeros(T, dtype=np.float64)
    cnt_post = np.zeros(T, dtype=np.int64)
    sum_t_post = np.zeros(T, dtype=np.float64)

    # Task4: treated only
    sum_y_tr = np.zeros(T, dtype=np.float64)
    cnt_tr = np.zeros(T, dtype=np.int64)
    sum_post_tr = np.zeros(T, dtype=np.float64)

    # Task5A: need means for y, post, treated_post
    sum_post_all = np.zeros(T, dtype=np.float64)
    sum_tp_all = np.zeros(T, dtype=np.float64)

    n_papers = 0
    n_papers_pre = 0
    n_papers_post = 0
    n_papers_tr = 0
    n_papers_did = 0

    for batch in pf.iter_batches(batch_size=200_000, columns=cols):
        b = batch
        treated = np.asarray(b.column("treated"))
        pub_year = np.asarray(pc.year(b.column("publicationdate")))
        ret_year = np.asarray(pc.year(b.column("RetractionDate")))

        y_mat = _vector_list_col_to_2d_fixed(b.column(outcome_col), offset=vector_offset, years=T)
        if transform == "log1p":
            y_mat = np.log1p(y_mat)

        fin = np.isfinite(y_mat)

        valid_y = fin.any(axis=1)
        n_papers_did += int(valid_y.sum())

        year_mat = pub_year[:, None].astype(np.int32) + ages[None, :].astype(np.int32) - 1
        post = (year_mat >= ret_year[:, None].astype(np.int32)).astype(np.int8)
        treated_i = treated.astype(np.int8)[:, None]
        tp = (treated_i * post).astype(np.int8)

        # All sample
        sum_y_all += np.nansum(y_mat, axis=0)
        cnt_all += fin.sum(axis=0, dtype=np.int64)
        sum_t_all += (treated_i * fin).sum(axis=0, dtype=np.int64)
        sum_post_all += (post * fin).sum(axis=0, dtype=np.int64)
        sum_tp_all += (tp * fin).sum(axis=0, dtype=np.int64)

        # Pre/post masks
        pre_m = (post == 0)
        post_m = (post == 1)
        fin_pre = fin & pre_m
        fin_post = fin & post_m

        sum_y_pre += np.nansum(y_mat * fin_pre, axis=0)
        cnt_pre += fin_pre.sum(axis=0, dtype=np.int64)
        sum_t_pre += (treated_i * fin_pre).sum(axis=0, dtype=np.int64)

        sum_y_post += np.nansum(y_mat * fin_post, axis=0)
        cnt_post += fin_post.sum(axis=0, dtype=np.int64)
        sum_t_post += (treated_i * fin_post).sum(axis=0, dtype=np.int64)

        # Treated-only sample for Task4
        tr_mask = (treated == 1)
        if tr_mask.any():
            y_tr = y_mat[tr_mask]
            post_tr = post[tr_mask]
            fin_tr = fin[tr_mask]
            sum_y_tr += np.nansum(y_tr, axis=0)
            cnt_tr += fin_tr.sum(axis=0, dtype=np.int64)
            sum_post_tr += (post_tr * fin_tr).sum(axis=0, dtype=np.int64)
            n_papers_tr += int((fin_tr.any(axis=1)).sum())

        n_papers += int(len(treated))
        n_papers_pre += int((fin_pre.sum(axis=1) > 0).sum())
        n_papers_post += int((fin_post.sum(axis=1) > 0).sum())

    # Means by age
    mean_y_all = sum_y_all / np.maximum(cnt_all, 1)
    mean_t_all = sum_t_all / np.maximum(cnt_all, 1)

    mean_y_pre = sum_y_pre / np.maximum(cnt_pre, 1)
    mean_t_pre = sum_t_pre / np.maximum(cnt_pre, 1)

    mean_y_post = sum_y_post / np.maximum(cnt_post, 1)
    mean_t_post = sum_t_post / np.maximum(cnt_post, 1)

    mean_y_tr = sum_y_tr / np.maximum(cnt_tr, 1)
    mean_post_tr = sum_post_tr / np.maximum(cnt_tr, 1)

    mean_post_all = sum_post_all / np.maximum(cnt_all, 1)
    mean_tp_all = sum_tp_all / np.maximum(cnt_all, 1)

    overall_mean_y = float(sum_y_all.sum() / np.maximum(cnt_all.sum(), 1))
    overall_mean_post = float(sum_post_all.sum() / np.maximum(cnt_all.sum(), 1))
    overall_mean_tp = float(sum_tp_all.sum() / np.maximum(cnt_all.sum(), 1))

    # ---------- Pass 1: estimate betas (using age-FE demeaning) ----------
    # Tasks 1-3: y ~ treated + age FE (1 regressor)
    num1 = 0.0
    den1 = 0.0
    num2 = 0.0
    den2 = 0.0
    num3 = 0.0
    den3 = 0.0

    # Task4: treated-only y ~ post + age FE
    num4 = 0.0
    den4 = 0.0

    # Task5A: two-way FE with paper FE + age FE for y ~ post + treated_post
    XtX5 = np.zeros((2, 2), dtype=np.float64)
    Xty5 = np.zeros(2, dtype=np.float64)
    nobs5 = 0

    for batch in pf.iter_batches(batch_size=200_000, columns=cols):
        b = batch
        treated = np.asarray(b.column("treated")).astype(np.int8)
        pub_year = np.asarray(pc.year(b.column("publicationdate")))
        ret_year = np.asarray(pc.year(b.column("RetractionDate")))
        y_mat = _vector_list_col_to_2d_fixed(b.column(outcome_col), offset=vector_offset, years=T)
        if transform == "log1p":
            y_mat = np.log1p(y_mat)

        fin = np.isfinite(y_mat)
        valid_y = fin.any(axis=1)

        year_mat = pub_year[:, None].astype(np.int32) + ages[None, :].astype(np.int32) - 1
        post = (year_mat >= ret_year[:, None].astype(np.int32)).astype(np.int8)
        treated_i = treated[:, None]
        tp = (treated_i * post).astype(np.int8)

        # Task1
        ytil = np.where(fin, y_mat - mean_y_all[None, :], 0.0)
        xtil = np.where(fin, treated_i.astype(np.float64) - mean_t_all[None, :], 0.0)
        num1 += float(np.sum(xtil * ytil))
        den1 += float(np.sum(xtil * xtil))

        # Task2 pre
        pre_m = (post == 0)
        if pre_m.any():
            fin_pre = fin & pre_m
            ytil2 = np.where(fin_pre, y_mat - mean_y_pre[None, :], 0.0)
            xtil2 = np.where(fin_pre, treated_i.astype(np.float64) - mean_t_pre[None, :], 0.0)
            num2 += float(np.sum(xtil2 * ytil2))
            den2 += float(np.sum(xtil2 * xtil2))

        # Task3 post
        post_m = (post == 1)
        if post_m.any():
            fin_post = fin & post_m
            ytil3 = np.where(fin_post, y_mat - mean_y_post[None, :], 0.0)
            xtil3 = np.where(fin_post, treated_i.astype(np.float64) - mean_t_post[None, :], 0.0)
            num3 += float(np.sum(xtil3 * ytil3))
            den3 += float(np.sum(xtil3 * xtil3))

        # Task4 treated-only
        tr_mask = (treated == 1)
        if tr_mask.any():
            y_tr = y_mat[tr_mask]
            post_tr = post[tr_mask].astype(np.float64)
            fin_tr = fin[tr_mask]
            ytil4 = np.where(fin_tr, y_tr - mean_y_tr[None, :], 0.0)
            xtil4 = np.where(fin_tr, post_tr - mean_post_tr[None, :], 0.0)
            num4 += float(np.sum(xtil4 * ytil4))
            den4 += float(np.sum(xtil4 * xtil4))

        # Task5A: paper FE + age FE => residualize by (paper, age) using additive demeaning
        if valid_y.any():
            y_use = y_mat[valid_y]
            post_use = post[valid_y]
            tp_use = tp[valid_y]

            fin_use = np.isfinite(y_use)

            y_i = np.nanmean(y_use, axis=1).astype(np.float64)
            post_f = post_use.astype(np.float64)
            tp_f = tp_use.astype(np.float64)
            post_i = post_f.mean(axis=1)
            tp_i = tp_f.mean(axis=1)

            y_res = y_use - y_i[:, None] - mean_y_all[None, :] + overall_mean_y
            post_res = post_f - post_i[:, None] - mean_post_all[None, :] + overall_mean_post
            tp_res = tp_f - tp_i[:, None] - mean_tp_all[None, :] + overall_mean_tp

            # Listwise deletion for missing outcomes: set (y, X) to 0 where y is missing
            y_res = np.where(fin_use, y_res, 0.0)
            post_res = np.where(fin_use, post_res, 0.0)
            tp_res = np.where(fin_use, tp_res, 0.0)

            x1 = post_res.reshape(-1)
            x2 = tp_res.reshape(-1)
            yy = y_res.reshape(-1)
            XtX5[0, 0] += float(np.dot(x1, x1))
            XtX5[0, 1] += float(np.dot(x1, x2))
            XtX5[1, 0] += float(np.dot(x2, x1))
            XtX5[1, 1] += float(np.dot(x2, x2))
            Xty5[0] += float(np.dot(x1, yy))
            Xty5[1] += float(np.dot(x2, yy))
            nobs5 += int(fin_use.sum())

    try:
        beta5 = np.linalg.solve(XtX5, Xty5)
    except np.linalg.LinAlgError:
        beta5 = np.linalg.lstsq(XtX5, Xty5, rcond=None)[0]
    meat5 = np.zeros((2, 2), dtype=np.float64)
    clusters5 = n_papers_did

    beta1 = num1 / den1 if den1 else float("nan")
    beta2 = num2 / den2 if den2 else float("nan")
    beta3 = num3 / den3 if den3 else float("nan")
    beta4 = num4 / den4 if den4 else float("nan")

    # ---------- Pass 2: cluster-robust SE (cluster by match_id/corpusid) ----------
    meat1 = 0.0
    meat2 = 0.0
    meat3 = 0.0
    meat4 = 0.0
    meat5 = np.zeros((2, 2), dtype=np.float64)

    nobs1 = int(cnt_all.sum())
    nobs2 = int(cnt_pre.sum())
    nobs3 = int(cnt_post.sum())
    nobs4 = int(cnt_tr.sum())
    clusters1 = n_papers
    clusters2 = n_papers_pre
    clusters3 = n_papers_post
    clusters4 = n_papers_tr
    clusters5 = n_papers_did

    for batch in pf.iter_batches(batch_size=200_000, columns=cols):
        b = batch
        treated = np.asarray(b.column("treated")).astype(np.int8)
        pub_year = np.asarray(pc.year(b.column("publicationdate")))
        ret_year = np.asarray(pc.year(b.column("RetractionDate")))
        y_mat = _vector_list_col_to_2d_fixed(b.column(outcome_col), offset=vector_offset, years=T)
        if transform == "log1p":
            y_mat = np.log1p(y_mat)

        fin = np.isfinite(y_mat)
        valid_y = fin.any(axis=1)

        year_mat = pub_year[:, None].astype(np.int32) + ages[None, :].astype(np.int32) - 1
        post = (year_mat >= ret_year[:, None].astype(np.int32)).astype(np.int8)
        treated_i = treated[:, None]
        tp = (treated_i * post).astype(np.int8)

        # Task1 score per paper
        ytil = np.where(fin, y_mat - mean_y_all[None, :], 0.0)
        xtil = np.where(fin, treated_i.astype(np.float64) - mean_t_all[None, :], 0.0)
        u = (ytil - beta1 * xtil)
        s = np.sum(xtil * u, axis=1)
        meat1 += float(np.dot(s, s))

        # Task2 pre
        pre_m = (post == 0)
        if pre_m.any():
            fin_pre = fin & pre_m
            ytil2 = np.where(fin_pre, y_mat - mean_y_pre[None, :], 0.0)
            xtil2 = np.where(fin_pre, treated_i.astype(np.float64) - mean_t_pre[None, :], 0.0)
            u2 = ytil2 - beta2 * xtil2
            s2 = np.sum(xtil2 * u2, axis=1)
            # papers with no pre obs have score 0
            meat2 += float(np.dot(s2, s2))

        # Task3 post
        post_m = (post == 1)
        if post_m.any():
            fin_post = fin & post_m
            ytil3 = np.where(fin_post, y_mat - mean_y_post[None, :], 0.0)
            xtil3 = np.where(fin_post, treated_i.astype(np.float64) - mean_t_post[None, :], 0.0)
            u3 = ytil3 - beta3 * xtil3
            s3 = np.sum(xtil3 * u3, axis=1)
            meat3 += float(np.dot(s3, s3))

        # Task4 treated-only
        tr_mask = (treated == 1)
        if tr_mask.any():
            y_tr = y_mat[tr_mask]
            post_tr = post[tr_mask].astype(np.float64)
            fin_tr = fin[tr_mask]
            ytil4 = np.where(fin_tr, y_tr - mean_y_tr[None, :], 0.0)
            xtil4 = np.where(fin_tr, post_tr - mean_post_tr[None, :], 0.0)
            u4 = ytil4 - beta4 * xtil4
            s4 = np.sum(xtil4 * u4, axis=1)
            meat4 += float(np.dot(s4, s4))

        # Task5A
        if valid_y.any():
            y_use = y_mat[valid_y]
            post_use = post[valid_y]
            tp_use = tp[valid_y]

            fin_use = np.isfinite(y_use)

            y_i = np.nanmean(y_use, axis=1).astype(np.float64)
            post_f = post_use.astype(np.float64)
            tp_f = tp_use.astype(np.float64)
            post_i = post_f.mean(axis=1)
            tp_i = tp_f.mean(axis=1)

            y_res = y_use - y_i[:, None] - mean_y_all[None, :] + overall_mean_y
            post_res = post_f - post_i[:, None] - mean_post_all[None, :] + overall_mean_post
            tp_res = tp_f - tp_i[:, None] - mean_tp_all[None, :] + overall_mean_tp

            y_res = np.where(fin_use, y_res, 0.0)
            post_res = np.where(fin_use, post_res, 0.0)
            tp_res = np.where(fin_use, tp_res, 0.0)

            u5 = y_res - (beta5[0] * post_res + beta5[1] * tp_res)
            s1 = np.sum(post_res * u5, axis=1)
            s2 = np.sum(tp_res * u5, axis=1)
            meat5[0, 0] += float(np.dot(s1, s1))
            meat5[0, 1] += float(np.dot(s1, s2))
            meat5[1, 0] += float(np.dot(s2, s1))
            meat5[1, 1] += float(np.dot(s2, s2))

    # Build results
    res1 = _cluster_robust_kreg(
        XtX=np.array([[den1]], dtype=float),
        Xty=np.array([num1], dtype=float),
        meat=np.array([[meat1]], dtype=float),
        nobs=nobs1,
        n_clusters=clusters1,
        coef_names=["treated"],
    )
    res2 = _cluster_robust_kreg(
        XtX=np.array([[den2]], dtype=float),
        Xty=np.array([num2], dtype=float),
        meat=np.array([[meat2]], dtype=float),
        nobs=nobs2,
        n_clusters=max(clusters2, 2),
        coef_names=["treated"],
    )
    res3 = _cluster_robust_kreg(
        XtX=np.array([[den3]], dtype=float),
        Xty=np.array([num3], dtype=float),
        meat=np.array([[meat3]], dtype=float),
        nobs=nobs3,
        n_clusters=max(clusters3, 2),
        coef_names=["treated"],
    )
    res4 = _cluster_robust_kreg(
        XtX=np.array([[den4]], dtype=float),
        Xty=np.array([num4], dtype=float),
        meat=np.array([[meat4]], dtype=float),
        nobs=nobs4,
        n_clusters=max(clusters4, 2),
        coef_names=["post_ret"],
    )
    res5 = _cluster_robust_kreg(
        XtX=XtX5,
        Xty=Xty5,
        meat=meat5,
        nobs=nobs5,
        n_clusters=clusters5,
        coef_names=["post_ret", "treated_post"],
    )

    outcome_label = outcome_col
    if transform == "log1p":
        outcome_label = f"log1p({outcome_label})"

    print(f"\n========== Distance = {distance_value} (stream) ==========")
    _print_result(f"Task 1: treated vs control ({outcome_label})", res1, cluster_label="corpusid (treated id)")
    _print_result(f"Task 2: pre-retraction treated vs control ({outcome_label})", res2, cluster_label="corpusid (treated id)")
    _print_result(f"Task 3: post-retraction treated vs control ({outcome_label})", res3, cluster_label="corpusid (treated id)")
    _print_result(f"Task 4: treated-only post vs pre ({outcome_label})", res4, cluster_label="corpusid (treated id)")

    note = "  DiD effect is coef on treated_post. Pretrend leads test skipped in streaming mode."
    if pretrend_leads >= 2:
        note = "  DiD effect is coef on treated_post. Pretrend leads test not available in streaming mode; use non-stream mode on a smaller sample." 
    _print_result(
        f"Task 5A (DiD): distance={distance_value} ({outcome_label})",
        res5,
        effect_note=note,
        cluster_label="corpusid (treated id)",
    )

    return res5


def _rowgroup_treated_split(pf) -> tuple[list[int], list[int]]:
    """Return (treated_row_groups, control_row_groups) indices.

    Assumes vector_current files are stored with treated rows first and controls later
    (true for the provided data_all/*.parquet). We detect this by checking each row group.
    """
    import pyarrow.compute as pc  # type: ignore

    treated_rgs: list[int] = []
    control_rgs: list[int] = []
    nrg = pf.metadata.num_row_groups
    for i in range(nrg):
        t = pf.read_row_group(i, columns=["treated"]).column("treated")
        # Row groups in your data are uniform (all 0 or all 1)
        mn = int(pc.min(t).as_py())
        mx = int(pc.max(t).as_py())
        if mn == 1 and mx == 1:
            treated_rgs.append(i)
        elif mn == 0 and mx == 0:
            control_rgs.append(i)
        else:
            # Mixed row group: rare. Treat as both by reading and splitting later.
            treated_rgs.append(i)
            control_rgs.append(i)
    return treated_rgs, control_rgs


def run_stream_vector_matched_1to1(
    paths: list[str],
    *,
    outcome_col: str,
    years: int,
    vector_offset: int,
    transform: str,
    distances: list[int] | None,
    time_fe: str,
    pretrend_leads: int,
    unified_5b: bool,
    base_distance: int | None,
    event_study: bool = False,
    es_leads: int = 3,
    es_lags: int = 5,
    event_study_csv: str | None = None,
) -> None:
    """Low-memory runner for 1:1 matched vector_current parquet.

    Assumptions (verified on provided data_all/*.parquet):
      - corpusid is match_id and appears exactly twice: treated=1 and treated=0
      - treated rows are stored first, controls later (by row group)
      - treated and control within match share publication year

    This implementation supports calendar-year time FE ("year") in a memory-safe way.
    """

    import pyarrow.parquet as pq  # type: ignore
    import pyarrow.compute as pc  # type: ignore

    if years < 1:
        raise ValueError("--years must be >= 1")
    if transform not in {"none", "log1p"}:
        raise ValueError("--transform must be 'none' or 'log1p'")

    # For reviewer-facing DiD, we use calendar year FE. If user requested year+age,
    # Tasks 1–3 and 5 are identical under 1:1 matching with identical pub_year,
    # and Tasks 4–5 would be collinear with paper FE; we therefore treat year+age as year.
    tfe = time_fe.strip().lower()
    if tfe not in {"year", "year+age"}:
        raise ValueError("Matched streaming mode currently supports --time-fe year (or year+age treated as year).")
    if tfe == "year+age":
        print("[note] --time-fe year+age requested; using calendar-year FE ('year') in matched streaming mode.")

    T = int(years)
    ages = np.arange(1, T + 1, dtype=np.int16)

    # Collect per-distance 5A results for optional streaming heterogeneity summary.
    res5a_by_d: dict[int, OLSResult] = {}

    # Unified 5B accumulators (paper FE + year FE, cluster by match_id)
    use_dist = distances

    # We need to know which distances exist in paths.
    inferred: list[tuple[int, str]] = []
    for p in paths:
        d = _infer_distance_from_path(p)
        if d is None:
            raise ValueError(f"Streaming matched mode expects per-distance filenames like data_1.parquet. Got: {p}")
        if use_dist is not None and d not in use_dist:
            continue
        inferred.append((int(d), p))

    if not inferred:
        raise ValueError("No distance files selected")

    inferred.sort(key=lambda x: x[0])
    run_ds = [d for d, _ in inferred]
    base = int(base_distance) if base_distance is not None else min(run_ds)
    if base not in run_ds:
        raise ValueError(f"--base-distance={base} not in distances being run: {run_ds}")

    # Unified regressor layout: [post, treated_post, tp_d? ...]
    het_ds = [d for d in run_ds if d != base]
    k_uni = 2 + len(het_ds)
    XtX_uni = np.zeros((k_uni, k_uni), dtype=np.float64)
    Xty_uni = np.zeros(k_uni, dtype=np.float64)
    meat_uni = np.zeros((k_uni, k_uni), dtype=np.float64)
    nobs_uni = 0
    n_clusters_uni = 0

    def _het_col_index(d: int) -> int | None:
        if d == base:
            return None
        return 2 + het_ds.index(d)

    # --- Unified 5B pooled year means (common calendar-year FE across distances) ---
    pooled_year_min: int | None = None
    pooled_year_max: int | None = None
    pooled_cnt: np.ndarray | None = None
    pooled_mean_y: np.ndarray | None = None
    pooled_mean_post: np.ndarray | None = None
    pooled_mean_tp: np.ndarray | None = None
    pooled_mean_het: dict[int, np.ndarray] = {}
    pooled_overall_y = pooled_overall_post = pooled_overall_tp = 0.0
    pooled_overall_het: dict[int, float] = {}

    if unified_5b:
        # Determine pooled calendar-year index range from publication years.
        pub_min = None
        pub_max = None
        for d0, path0 in inferred:
            pf0 = pq.ParquetFile(path0)
            for batch0 in pf0.iter_batches(batch_size=200_000, columns=["publicationdate"]):
                pub_y0 = np.asarray(pc.year(batch0.column("publicationdate"))).astype(np.int32)
                if pub_y0.size == 0:
                    continue
                mn = int(pub_y0.min())
                mx = int(pub_y0.max())
                pub_min = mn if pub_min is None else min(pub_min, mn)
                pub_max = mx if pub_max is None else max(pub_max, mx)
        if pub_min is None or pub_max is None:
            raise ValueError("Unified 5B: could not infer publication-year range")
        pooled_year_min = int(pub_min)
        pooled_year_max = int(pub_max + (T - 1))
        YY = int(pooled_year_max - pooled_year_min + 1)
        if YY <= 0:
            raise ValueError("Unified 5B: invalid pooled calendar-year range")

        cnt_y = np.zeros(YY, dtype=np.int64)
        sum_y = np.zeros(YY, dtype=np.float64)
        sum_post = np.zeros(YY, dtype=np.float64)
        sum_tp = np.zeros(YY, dtype=np.float64)
        sum_het: dict[int, np.ndarray] = {int(dd): np.zeros(YY, dtype=np.float64) for dd in het_ds}

        cols0 = ["treated", "publicationdate", "RetractionDate", outcome_col]
        for d0, path0 in inferred:
            pf0 = pq.ParquetFile(path0)
            d0i = int(d0)
            for batch0 in pf0.iter_batches(batch_size=200_000, columns=cols0):
                treated0 = np.asarray(batch0.column("treated")).astype(np.int8)
                pub_y0 = np.asarray(pc.year(batch0.column("publicationdate"))).astype(np.int32)
                ret_y0 = np.asarray(pc.year(batch0.column("RetractionDate"))).astype(np.int32)

                y_mat0 = _vector_list_col_to_2d_fixed(batch0.column(outcome_col), offset=vector_offset, years=T)
                if transform == "log1p":
                    y_mat0 = np.log1p(y_mat0)
                fin0 = np.isfinite(y_mat0)

                year_mat0 = pub_y0[:, None] + ages[None, :].astype(np.int32) - 1
                post0 = (year_mat0 >= ret_y0[:, None]).astype(np.float64)
                tp0 = post0 * treated0.astype(np.float64)[:, None]

                for j0 in range(T):
                    m0 = fin0[:, j0]
                    if not m0.any():
                        continue
                    yy0 = (year_mat0[m0, j0] - int(pooled_year_min)).astype(np.int32)
                    np.add.at(cnt_y, yy0, 1)
                    np.add.at(sum_y, yy0, y_mat0[m0, j0])
                    np.add.at(sum_post, yy0, post0[m0, j0])
                    np.add.at(sum_tp, yy0, tp0[m0, j0])
                    if d0i in sum_het:
                        np.add.at(sum_het[d0i], yy0, tp0[m0, j0])

        if int(cnt_y.sum()) <= 0:
            raise ValueError("Unified 5B: no usable observations to compute pooled year means")

        pooled_cnt = cnt_y
        pooled_mean_y = sum_y / np.maximum(cnt_y, 1)
        pooled_mean_post = sum_post / np.maximum(cnt_y, 1)
        pooled_mean_tp = sum_tp / np.maximum(cnt_y, 1)
        pooled_overall_y = float(sum_y.sum() / cnt_y.sum())
        pooled_overall_post = float(sum_post.sum() / cnt_y.sum())
        pooled_overall_tp = float(sum_tp.sum() / cnt_y.sum())

        for dd in het_ds:
            ddi = int(dd)
            pooled_mean_het[ddi] = sum_het[ddi] / np.maximum(cnt_y, 1)
            pooled_overall_het[ddi] = float(sum_het[ddi].sum() / cnt_y.sum())

        # --- Unified 5B overlap-aware treated-paper means across distances ---
        # When the same treated corpusid appears in multiple distance files, the pooled
        # paper FE demeaning must use means computed over *all* distances (the pooled sample),
        # not within a single distance file. Controls are distance-specific by construction.
        print("[note] Unified 5B: computing overlap-aware treated-paper means across distances...")

        treated_ids_all: np.ndarray | None = None
        for _d0, _p0 in inferred:
            _pf0 = pq.ParquetFile(_p0)
            _treated_rgs0, _ = _rowgroup_treated_split(_pf0)
            _chunks: list[np.ndarray] = []
            for _i0 in _treated_rgs0:
                _rg0 = _pf0.read_row_group(_i0, columns=["corpusid", "treated"])
                _tr0 = np.asarray(_rg0.column("treated")).astype(np.int8)
                _m0 = (_tr0 == 1)
                if not _m0.any():
                    continue
                _chunks.append(np.asarray(_rg0.column("corpusid"))[_m0].astype(np.int64, copy=False))
            if not _chunks:
                continue
            _ids0 = np.concatenate(_chunks)
            _ids0.sort()
            if _ids0.size >= 2 and np.any(_ids0[1:] == _ids0[:-1]):
                _ids0 = np.unique(_ids0)
            treated_ids_all = _ids0 if treated_ids_all is None else np.union1d(treated_ids_all, _ids0)

        if treated_ids_all is None or treated_ids_all.size == 0:
            raise ValueError("Unified 5B: could not collect treated paper IDs")

        n_ids_all = int(treated_ids_all.size)
        treated_cnt_all = np.zeros(n_ids_all, dtype=np.int32)
        treated_sum_y_all = np.zeros(n_ids_all, dtype=np.float64)
        treated_sum_post_all = np.zeros(n_ids_all, dtype=np.float64)
        treated_sum_het_all: dict[int, np.ndarray] = {int(dd): np.zeros(n_ids_all, dtype=np.float64) for dd in het_ds}

        cols_tm = ["corpusid", "treated", "publicationdate", "RetractionDate", outcome_col]
        for _d0, _p0 in inferred:
            _pf0 = pq.ParquetFile(_p0)
            _d0i = int(_d0)
            for _batch0 in _pf0.iter_batches(batch_size=200_000, columns=cols_tm):
                _tr0 = np.asarray(_batch0.column("treated")).astype(np.int8)
                _m0 = (_tr0 == 1)
                if not _m0.any():
                    continue
                _mid0 = np.asarray(_batch0.column("corpusid"))[_m0].astype(np.int64)
                _pos0 = np.searchsorted(treated_ids_all, _mid0)
                if _pos0.size == 0:
                    continue
                if int(_pos0.max()) >= n_ids_all or not np.array_equal(treated_ids_all[_pos0], _mid0):
                    raise ValueError("Unified 5B: treated ID mapping failed while computing overlap-aware means")

                _pub_y0 = np.asarray(pc.year(_batch0.column("publicationdate")))[_m0].astype(np.int32)
                _ret_y0 = np.asarray(pc.year(_batch0.column("RetractionDate")))[_m0].astype(np.int32)
                _y0 = _vector_list_col_to_2d_fixed(_batch0.column(outcome_col), offset=vector_offset, years=T)[_m0].astype(
                    np.float64
                )
                if transform == "log1p":
                    _y0 = np.log1p(_y0)
                _fin0 = np.isfinite(_y0)
                _cnt0 = _fin0.sum(axis=1).astype(np.int32)
                if int(_cnt0.sum()) <= 0:
                    continue

                _year0 = _pub_y0[:, None] + ages[None, :].astype(np.int32) - 1
                _post0 = (_year0 >= _ret_y0[:, None])
                _post_sum0 = (_post0 & _fin0).sum(axis=1).astype(np.float64)

                treated_cnt_all[_pos0] += _cnt0
                treated_sum_y_all[_pos0] += np.nansum(_y0, axis=1)
                treated_sum_post_all[_pos0] += _post_sum0
                if _d0i in treated_sum_het_all:
                    treated_sum_het_all[_d0i][_pos0] += _post_sum0

    # Process each distance file sequentially (low memory).
    es_rows_out: list[dict[str, float | int | str]] = []
    for d, path in inferred:
        cols = ["corpusid", "treated", "publicationdate", "RetractionDate", outcome_col]
        pf = pq.ParquetFile(path)
        schema_cols = set(pf.schema_arrow.names)
        missing = [c for c in cols if c not in schema_cols]
        if missing:
            raise ValueError(f"Vector format missing columns {missing} in file {path}")

        treated_rgs, control_rgs = _rowgroup_treated_split(pf)
        if not treated_rgs or not control_rgs:
            raise ValueError(f"Expected both treated and control rows in {path}")

        # ---------- Load treated block into memory (match-aligned arrays) ----------
        n_treated = int(sum(pf.metadata.row_group(i).num_rows for i in treated_rgs))
        ids = np.empty(n_treated, dtype=np.int64)
        pub = np.empty(n_treated, dtype=np.int16)
        ret = np.empty(n_treated, dtype=np.int16)
        y_t = np.empty((n_treated, T), dtype=np.float32)
        write = 0

        for i in treated_rgs:
            rg = pf.read_row_group(i, columns=cols)
            tr = np.asarray(rg.column("treated")).astype(np.int8)
            # allow for rare mixed row group
            mask = (tr == 1)
            if not mask.any():
                continue
            mid = np.asarray(rg.column("corpusid")).astype(np.int64)[mask]
            pub_y = np.asarray(pc.year(rg.column("publicationdate")))[mask].astype(np.int16)
            ret_y = np.asarray(pc.year(rg.column("RetractionDate")))[mask].astype(np.int16)
            y_mat = _vector_list_col_to_2d_fixed(rg.column(outcome_col), offset=vector_offset, years=T)[mask].astype(np.float32)
            if transform == "log1p":
                y_mat = np.log1p(y_mat)

            n = len(mid)
            ids[write : write + n] = mid
            pub[write : write + n] = pub_y
            ret[write : write + n] = ret_y
            y_t[write : write + n, :] = y_mat
            write += n

        if write != n_treated:
            ids = ids[:write]
            pub = pub[:write]
            ret = ret[:write]
            y_t = y_t[:write, :]
            n_treated = write

        order = np.argsort(ids)
        ids_s = ids[order]
        pub_s = pub[order].astype(np.int32)
        ret_s = ret[order].astype(np.int32)
        y_t = y_t[order, :].astype(np.float64)

        # Guardrails: ensure treated block has unique match IDs.
        # If this fails, any searchsorted-based alignment below would silently mispair.
        if ids_s.size >= 2 and np.any(ids_s[1:] == ids_s[:-1]):
            raise ValueError(
                f"Matched streaming expects each match_id (corpusid) to appear exactly once in treated rows. "
                f"Found duplicates in treated block for file: {path}"
            )

        def _idx_for(mid: np.ndarray) -> np.ndarray:
            """Map control match_ids to treated-row indices with strict validation."""
            idx0 = np.searchsorted(ids_s, mid)
            n0 = int(ids_s.shape[0])
            oob = idx0 >= n0
            if oob.any():
                bad_ids = mid[oob][:5]
                raise ValueError(
                    f"Matched streaming alignment failed: {int(oob.sum())} control match_id(s) not found in treated block "
                    f"for file {path}. Examples: {bad_ids.tolist()}"
                )
            got = ids_s[idx0]
            bad = got != mid
            if bad.any():
                ex = np.column_stack([mid[bad][:5], got[bad][:5]]).tolist()
                raise ValueError(
                    f"Matched streaming alignment failed: control match_id(s) did not match treated IDs after lookup "
                    f"for file {path}. Examples [requested, found]: {ex}"
                )
            return idx0

        # pre-compute post indicator for treated (same as control within match)
        year_mat = pub_s[:, None] + ages[None, :].astype(np.int32) - 1
        post = (year_mat >= ret_s[:, None]).astype(np.int8)
        post_f = post.astype(np.float64)

        tau_mat = year_mat - ret_s[:, None].astype(np.int32)

        # ---------- Pass A: compute Tasks 1–3 via within-match differences ----------
        sum_all = 0.0
        cnt_all = 0
        sum_pre = 0.0
        cnt_pre = 0
        sum_post = 0.0
        cnt_post = 0

        for i in control_rgs:
            rg = pf.read_row_group(i, columns=cols)
            tr = np.asarray(rg.column("treated")).astype(np.int8)
            mask0 = (tr == 0)
            if not mask0.any():
                continue
            mid = np.asarray(rg.column("corpusid")).astype(np.int64)[mask0]
            idx = _idx_for(mid)
            pub_y_c = np.asarray(pc.year(rg.column("publicationdate")))[mask0].astype(np.int32)
            if not np.array_equal(pub_y_c, pub_s[idx]):
                raise ValueError(
                    f"Matched streaming assumption violated: treated/control within a match must share publication year. "
                    f"File: {path}"
                )
            y_c = _vector_list_col_to_2d_fixed(rg.column(outcome_col), offset=vector_offset, years=T)[mask0].astype(np.float64)
            if transform == "log1p":
                y_c = np.log1p(y_c)

            dy = y_t[idx, :] - y_c
            fin = np.isfinite(dy)
            sum_all += float(np.nansum(dy))
            cnt_all += int(fin.sum())
            pre_m = (post[idx, :] == 0) & fin
            post_m = (post[idx, :] == 1) & fin
            sum_pre += float(dy[pre_m].sum())
            cnt_pre += int(pre_m.sum())
            sum_post += float(dy[post_m].sum())
            cnt_post += int(post_m.sum())

        # ---------- Optional event-study (matched stream): dy(tau) - dy(-1) ----------
        if event_study:
            L = int(es_leads)
            R = int(es_lags)
            tau_vals = [t for t in range(-L, R + 1) if t != -1]
            # Accumulate per-tau match-level diffs d_m(tau) = dy_m(tau) - dy_m(-1)
            sum_d = {int(tt): 0.0 for tt in tau_vals}
            sumsq_d = {int(tt): 0.0 for tt in tau_vals}
            G_d = {int(tt): 0 for tt in tau_vals}

            lead_taus = [-(k) for k in range(2, L + 1)]  # e.g., [-2, -3, ...]
            m_lead = len(lead_taus)
            sum_lead = np.zeros(m_lead, dtype=np.float64)
            sum_outer = np.zeros((m_lead, m_lead), dtype=np.float64)
            G_lead = 0

            for i in control_rgs:
                rg = pf.read_row_group(i, columns=cols)
                tr = np.asarray(rg.column("treated")).astype(np.int8)
                mask0 = (tr == 0)
                if not mask0.any():
                    continue
                mid = np.asarray(rg.column("corpusid")).astype(np.int64)[mask0]
                idx = _idx_for(mid)
                y_c = _vector_list_col_to_2d_fixed(rg.column(outcome_col), offset=vector_offset, years=T)[mask0].astype(np.float64)
                if transform == "log1p":
                    y_c = np.log1p(y_c)

                dy = y_t[idx, :] - y_c
                fin = np.isfinite(dy)
                tsub = tau_mat[idx, :]

                base_mask = (tsub == -1) & fin
                has_base = base_mask.any(axis=1)
                if not has_base.any():
                    continue
                dy_base = np.where(base_mask, dy, 0.0).sum(axis=1)

                # Per-tau means/SE
                for tt in tau_vals:
                    mask_tt = (tsub == int(tt)) & fin
                    has_tt = mask_tt.any(axis=1) & has_base
                    if not has_tt.any():
                        continue
                    dy_tt = np.where(mask_tt, dy, 0.0).sum(axis=1)
                    dvec = (dy_tt - dy_base)[has_tt]
                    sum_d[int(tt)] += float(dvec.sum())
                    sumsq_d[int(tt)] += float(np.dot(dvec, dvec))
                    G_d[int(tt)] += int(len(dvec))

                # Joint leads Wald on common sample having all leads and baseline
                if m_lead > 0:
                    has_all = has_base.copy()
                    D = np.zeros((len(idx), m_lead), dtype=np.float64)
                    for j, lt in enumerate(lead_taus):
                        mask_lt = (tsub == int(lt)) & fin
                        has_lt = mask_lt.any(axis=1)
                        has_all &= has_lt
                        dy_lt = np.where(mask_lt, dy, 0.0).sum(axis=1)
                        D[:, j] = dy_lt - dy_base
                    if has_all.any():
                        D_use = D[has_all, :]
                        sum_lead += D_use.sum(axis=0)
                        sum_outer += D_use.T @ D_use
                        G_lead += int(D_use.shape[0])

            # Print event-study table for this distance
            rows_es: list[tuple[int, float, float, float, float, float, float, int]] = []
            for tt in sorted(tau_vals):
                G = int(G_d[int(tt)])
                if G < 5:
                    continue
                est = float(sum_d[int(tt)] / G)
                # Var(mean) with CR1 for cluster-as-sample: ssd/(G*(G-1))
                ssd = float(sumsq_d[int(tt)] - (sum_d[int(tt)] * sum_d[int(tt)]) / G)
                var = ssd / (G * max(G - 1, 1))
                se = math.sqrt(var) if var >= 0 else float("nan")
                tstat = est / se if se else float("nan")
                stats = _require_scipy_stats()
                df = G - 1
                if np.isfinite(tstat):
                    p = float(2.0 * stats.t.sf(abs(tstat), df=df))
                    p = _clip_p01(p)
                    tcrit = float(stats.t.ppf(0.975, df=df))
                else:
                    p = float("nan")
                    tcrit = float("nan")
                lo = est - tcrit * se
                hi = est + tcrit * se
                rows_es.append((int(tt), est, se, float(tstat), float(p), float(lo), float(hi), int(G)))
                es_rows_out.append(
                    {
                        "distance": int(d),
                        "tau": int(tt),
                        "est": float(est),
                        "se": float(se),
                        "t": float(tstat),
                        "p": float(p),
                        "ci_low": float(lo),
                        "ci_high": float(hi),
                        "clusters": int(G),
                        "mode": "matched_stream_dy",
                    }
                )

            # Joint pretrend leads Wald
            msg_es = "Pretrend leads Wald: NA"
            if m_lead > 0 and G_lead >= 5:
                beta = (sum_lead / G_lead).reshape(-1, 1)
                S = sum_outer - float(G_lead) * (beta @ beta.T)
                V = S / (G_lead * max(G_lead - 1, 1))
                try:
                    try:
                        stat = float((beta.T @ np.linalg.solve(V, beta)).reshape(()))
                    except np.linalg.LinAlgError:
                        stat = float((beta.T @ (np.linalg.pinv(V) @ beta)).reshape(()))
                    df0 = int(m_lead)
                    p0, used_approx = _chi2_p_value(stat, df0)
                    msg_es = (
                        f"Pretrend leads Wald: chi2({df0})={stat:.3f}, p={_format_chi2_p(p0, used_approx)} "
                        f"(common sample G={G_lead:,})"
                    )
                except Exception:
                    msg_es = "Pretrend leads Wald: failed (singular V)"

            _print_event_study_table(
                f"Event-study (matched stream dy): distance={d} (ref tau=-1). {msg_es}",
                rows_es,
            )

        beta1 = sum_all / cnt_all if cnt_all else float("nan")
        beta2 = sum_pre / cnt_pre if cnt_pre else float("nan")
        beta3 = sum_post / cnt_post if cnt_post else float("nan")

        # ---------- Pass B: cluster-robust SE for Tasks 1–3 (cluster=corpusid treated id) ----------
        meat1 = 0.0
        meat2 = 0.0
        meat3 = 0.0
        n_clusters = int(n_treated)

        for i in control_rgs:
            rg = pf.read_row_group(i, columns=cols)
            tr = np.asarray(rg.column("treated")).astype(np.int8)
            mask0 = (tr == 0)
            if not mask0.any():
                continue
            mid = np.asarray(rg.column("corpusid")).astype(np.int64)[mask0]
            idx = _idx_for(mid)
            y_c = _vector_list_col_to_2d_fixed(rg.column(outcome_col), offset=vector_offset, years=T)[mask0].astype(np.float64)
            if transform == "log1p":
                y_c = np.log1p(y_c)
            dy = y_t[idx, :] - y_c
            fin = np.isfinite(dy)
            # scores per match (intercept-only on dy)
            r1 = np.where(fin, dy - beta1, 0.0).sum(axis=1)
            meat1 += float(np.dot(r1, r1))

            pre_m = (post[idx, :] == 0) & fin
            r2 = np.where(pre_m, dy - beta2, 0.0).sum(axis=1)
            meat2 += float(np.dot(r2, r2))

            post_m = (post[idx, :] == 1) & fin
            r3 = np.where(post_m, dy - beta3, 0.0).sum(axis=1)
            meat3 += float(np.dot(r3, r3))

        res1 = _cluster_robust_kreg(
            XtX=np.array([[float(cnt_all)]], dtype=float),
            Xty=np.array([float(sum_all)], dtype=float),
            meat=np.array([[float(meat1)]], dtype=float),
            nobs=int(cnt_all),
            n_clusters=n_clusters,
            coef_names=["treated"],
        )
        res2 = _cluster_robust_kreg(
            XtX=np.array([[float(cnt_pre)]], dtype=float),
            Xty=np.array([float(sum_pre)], dtype=float),
            meat=np.array([[float(meat2)]], dtype=float),
            nobs=int(cnt_pre),
            n_clusters=max(n_clusters, 2),
            coef_names=["treated"],
        )
        res3 = _cluster_robust_kreg(
            XtX=np.array([[float(cnt_post)]], dtype=float),
            Xty=np.array([float(sum_post)], dtype=float),
            meat=np.array([[float(meat3)]], dtype=float),
            nobs=int(cnt_post),
            n_clusters=max(n_clusters, 2),
            coef_names=["treated"],
        )

        # ---------- Task 4: treated-only post vs pre with paper FE + year FE ----------
        # Build year index range
        year_min = int(year_mat.min())
        year_max = int(year_mat.max())
        Y = year_max - year_min + 1
        # Means by calendar year (treated-only sample)
        sum_y_y = np.zeros(Y, dtype=np.float64)
        cnt_y_y = np.zeros(Y, dtype=np.int64)
        sum_p_y = np.zeros(Y, dtype=np.float64)
        cnt_p_y = np.zeros(Y, dtype=np.int64)

        y_fin = np.isfinite(y_t)
        for j in range(T):
            yy = year_mat[:, j] - year_min
            m = y_fin[:, j]
            if not m.any():
                continue
            np.add.at(sum_y_y, yy[m], y_t[m, j])
            np.add.at(cnt_y_y, yy[m], 1)
            np.add.at(sum_p_y, yy[m], post_f[m, j])
            np.add.at(cnt_p_y, yy[m], 1)

        mean_y_y = sum_y_y / np.maximum(cnt_y_y, 1)
        mean_p_y = sum_p_y / np.maximum(cnt_p_y, 1)
        overall_y = float(sum_y_y.sum() / np.maximum(cnt_y_y.sum(), 1))
        overall_p = float(sum_p_y.sum() / np.maximum(cnt_p_y.sum(), 1))

        # Safe means without RuntimeWarning for all-missing rows.
        y_cnt = np.isfinite(y_t).sum(axis=1).astype(np.float64)
        y_i = np.nansum(y_t, axis=1) / np.maximum(y_cnt, 1.0)
        # Paper mean of post must be computed on the estimation sample (finite y only).
        p_i = np.divide(
            (post_f * y_fin.astype(np.float64)).sum(axis=1),
            y_cnt,
            out=np.zeros_like(y_cnt),
            where=(y_cnt > 0),
        )

        num4 = 0.0
        den4 = 0.0
        s4_meat = 0.0
        for j in range(T):
            yy = year_mat[:, j] - year_min
            m = y_fin[:, j]
            if not m.any():
                continue
            y_res = y_t[m, j] - y_i[m] - mean_y_y[yy[m]] + overall_y
            x_res = post_f[m, j] - p_i[m] - mean_p_y[yy[m]] + overall_p
            num4 += float(np.dot(x_res, y_res))
            den4 += float(np.dot(x_res, x_res))

        beta4 = num4 / den4 if den4 else float("nan")

        # cluster scores (match_id == treated paper's match)
        # score per match: sum_t x_res * u
        scores4 = np.zeros(n_treated, dtype=np.float64)
        for j in range(T):
            yy = year_mat[:, j] - year_min
            m = y_fin[:, j]
            if not m.any():
                continue
            idx_m = np.flatnonzero(m)
            y_res = y_t[m, j] - y_i[m] - mean_y_y[yy[m]] + overall_y
            x_res = post_f[m, j] - p_i[m] - mean_p_y[yy[m]] + overall_p
            u = y_res - beta4 * x_res
            scores4[idx_m] += x_res * u
        s4_meat = float(np.dot(scores4, scores4))

        res4 = _cluster_robust_kreg(
            XtX=np.array([[float(den4)]], dtype=float),
            Xty=np.array([float(num4)], dtype=float),
            meat=np.array([[float(s4_meat)]], dtype=float),
            nobs=int(y_fin.sum()),
            n_clusters=n_clusters,
            coef_names=["post_ret"],
        )

        # ---------- Task 5A: DiD per distance with paper FE + year FE, cluster=corpusid treated id ----------
        # Compute year means for y, post, treated_post across BOTH treated and control papers.
        sum_y = np.zeros(Y, dtype=np.float64)
        cnt_y = np.zeros(Y, dtype=np.int64)
        sum_post = np.zeros(Y, dtype=np.float64)
        cnt_post = np.zeros(Y, dtype=np.int64)
        sum_tp = np.zeros(Y, dtype=np.float64)
        cnt_tp = np.zeros(Y, dtype=np.int64)

        # treated contribution
        for j in range(T):
            yy = year_mat[:, j] - year_min
            m = y_fin[:, j]
            if not m.any():
                continue
            np.add.at(sum_y, yy[m], y_t[m, j])
            np.add.at(cnt_y, yy[m], 1)
            np.add.at(sum_post, yy[m], post_f[m, j])
            np.add.at(cnt_post, yy[m], 1)
            np.add.at(sum_tp, yy[m], post_f[m, j])
            np.add.at(cnt_tp, yy[m], 1)

        # control contribution (stream row groups)
        for i in control_rgs:
            rg = pf.read_row_group(i, columns=cols)
            tr = np.asarray(rg.column("treated")).astype(np.int8)
            mask0 = (tr == 0)
            if not mask0.any():
                continue
            mid = np.asarray(rg.column("corpusid")).astype(np.int64)[mask0]
            idx = _idx_for(mid)
            y_c = _vector_list_col_to_2d_fixed(rg.column(outcome_col), offset=vector_offset, years=T)[mask0].astype(np.float64)
            if transform == "log1p":
                y_c = np.log1p(y_c)
            finc = np.isfinite(y_c)
            for j in range(T):
                yy = year_mat[idx, j] - year_min
                m = finc[:, j]
                if not m.any():
                    continue
                np.add.at(sum_y, yy[m], y_c[m, j])
                np.add.at(cnt_y, yy[m], 1)
                np.add.at(sum_post, yy[m], post_f[idx[m], j])
                np.add.at(cnt_post, yy[m], 1)
                # treated_post is 0 for controls
                np.add.at(sum_tp, yy[m], 0.0)
                np.add.at(cnt_tp, yy[m], 1)

        mean_y = sum_y / np.maximum(cnt_y, 1)
        mean_post = sum_post / np.maximum(cnt_post, 1)
        mean_tp = sum_tp / np.maximum(cnt_tp, 1)
        overall_y2 = float(sum_y.sum() / np.maximum(cnt_y.sum(), 1))
        overall_post2 = float(sum_post.sum() / np.maximum(cnt_post.sum(), 1))
        overall_tp2 = float(sum_tp.sum() / np.maximum(cnt_tp.sum(), 1))

        # Paper means for treated papers
        y_cnt2 = np.isfinite(y_t).sum(axis=1).astype(np.float64)
        y_it = np.nansum(y_t, axis=1) / np.maximum(y_cnt2, 1.0)
        post_it = np.divide(
            (post_f * y_fin.astype(np.float64)).sum(axis=1),
            y_cnt2,
            out=np.zeros_like(y_cnt2),
            where=(y_cnt2 > 0),
        )
        tp_it = post_it  # treated_post == post for treated

        XtX5 = np.zeros((2, 2), dtype=np.float64)
        Xty5 = np.zeros(2, dtype=np.float64)
        nobs5 = 0

        # Accumulate treated paper contributions
        for j in range(T):
            yy = year_mat[:, j] - year_min
            m = y_fin[:, j]
            if not m.any():
                continue
            y_res = y_t[m, j] - y_it[m] - mean_y[yy[m]] + overall_y2
            x_post = post_f[m, j] - post_it[m] - mean_post[yy[m]] + overall_post2
            x_tp = post_f[m, j] - tp_it[m] - mean_tp[yy[m]] + overall_tp2
            XtX5[0, 0] += float(np.dot(x_post, x_post))
            XtX5[0, 1] += float(np.dot(x_post, x_tp))
            XtX5[1, 0] += float(np.dot(x_tp, x_post))
            XtX5[1, 1] += float(np.dot(x_tp, x_tp))
            Xty5[0] += float(np.dot(x_post, y_res))
            Xty5[1] += float(np.dot(x_tp, y_res))
            nobs5 += int(m.sum())

        # Accumulate control paper contributions (stream)
        for i in control_rgs:
            rg = pf.read_row_group(i, columns=cols)
            tr = np.asarray(rg.column("treated")).astype(np.int8)
            mask0 = (tr == 0)
            if not mask0.any():
                continue
            mid = np.asarray(rg.column("corpusid")).astype(np.int64)[mask0]
            idx = _idx_for(mid)
            y_c = _vector_list_col_to_2d_fixed(rg.column(outcome_col), offset=vector_offset, years=T)[mask0].astype(np.float64)
            if transform == "log1p":
                y_c = np.log1p(y_c)
            finc = np.isfinite(y_c)
            y_cntc = np.isfinite(y_c).sum(axis=1).astype(np.float64)
            y_ic = np.nansum(y_c, axis=1) / np.maximum(y_cntc, 1.0)
            post_ic = np.divide(
                (post_f[idx, :] * finc.astype(np.float64)).sum(axis=1),
                y_cntc,
                out=np.zeros_like(y_cntc),
                where=(y_cntc > 0),
            )
            tp_ic = np.zeros_like(post_ic)
            for j in range(T):
                yy = year_mat[idx, j] - year_min
                m = finc[:, j]
                if not m.any():
                    continue
                y_res = y_c[m, j] - y_ic[m] - mean_y[yy[m]] + overall_y2
                x_post = post_f[idx[m], j] - post_ic[m] - mean_post[yy[m]] + overall_post2
                x_tp = 0.0 - tp_ic[m] - mean_tp[yy[m]] + overall_tp2
                XtX5[0, 0] += float(np.dot(x_post, x_post))
                XtX5[0, 1] += float(np.dot(x_post, x_tp))
                XtX5[1, 0] += float(np.dot(x_tp, x_post))
                XtX5[1, 1] += float(np.dot(x_tp, x_tp))
                Xty5[0] += float(np.dot(x_post, y_res))
                Xty5[1] += float(np.dot(x_tp, y_res))
                nobs5 += int(m.sum())

        try:
            beta5 = np.linalg.solve(XtX5, Xty5)
        except np.linalg.LinAlgError:
            beta5 = np.linalg.lstsq(XtX5, Xty5, rcond=None)[0]

        # Meat for cluster-robust (match) in one control pass: each match computed once.
        meat5 = np.zeros((2, 2), dtype=np.float64)
        for i in control_rgs:
            rg = pf.read_row_group(i, columns=cols)
            tr = np.asarray(rg.column("treated")).astype(np.int8)
            mask0 = (tr == 0)
            if not mask0.any():
                continue
            mid = np.asarray(rg.column("corpusid")).astype(np.int64)[mask0]
            idx = _idx_for(mid)
            y_c = _vector_list_col_to_2d_fixed(rg.column(outcome_col), offset=vector_offset, years=T)[mask0].astype(np.float64)
            if transform == "log1p":
                y_c = np.log1p(y_c)
            finc = np.isfinite(y_c)
            y_cntc2 = np.isfinite(y_c).sum(axis=1).astype(np.float64)
            y_ic = np.nansum(y_c, axis=1) / np.maximum(y_cntc2, 1.0)
            post_ic = np.divide(
                (post_f[idx, :] * finc.astype(np.float64)).sum(axis=1),
                y_cntc2,
                out=np.zeros_like(y_cntc2),
                where=(y_cntc2 > 0),
            )
            tp_ic = np.zeros_like(post_ic)

            # treated paper stats for these matches
            yt = y_t[idx, :]
            fint = np.isfinite(yt)
            y_it_b = y_it[idx]
            post_it_b = post_it[idx]
            tp_it_b = tp_it[idx]

            s1 = np.zeros(len(idx), dtype=np.float64)
            s2 = np.zeros(len(idx), dtype=np.float64)

            for j in range(T):
                yy = year_mat[idx, j] - year_min

                # treated contribution
                mt = fint[:, j]
                if mt.any():
                    y_res = yt[mt, j] - y_it_b[mt] - mean_y[yy[mt]] + overall_y2
                    x_post = post_f[idx[mt], j] - post_it_b[mt] - mean_post[yy[mt]] + overall_post2
                    x_tp = post_f[idx[mt], j] - tp_it_b[mt] - mean_tp[yy[mt]] + overall_tp2
                    u = y_res - (beta5[0] * x_post + beta5[1] * x_tp)
                    s1[mt] += x_post * u
                    s2[mt] += x_tp * u

                # control contribution
                mc = finc[:, j]
                if mc.any():
                    y_res = y_c[mc, j] - y_ic[mc] - mean_y[yy[mc]] + overall_y2
                    x_post = post_f[idx[mc], j] - post_ic[mc] - mean_post[yy[mc]] + overall_post2
                    x_tp = 0.0 - tp_ic[mc] - mean_tp[yy[mc]] + overall_tp2
                    u = y_res - (beta5[0] * x_post + beta5[1] * x_tp)
                    s1[mc] += x_post * u
                    s2[mc] += x_tp * u

            meat5[0, 0] += float(np.dot(s1, s1))
            meat5[0, 1] += float(np.dot(s1, s2))
            meat5[1, 0] += float(np.dot(s2, s1))
            meat5[1, 1] += float(np.dot(s2, s2))

        res5 = _cluster_robust_kreg(
            XtX=XtX5,
            Xty=Xty5,
            meat=meat5,
            nobs=nobs5,
            n_clusters=n_clusters,
            coef_names=["post_ret", "treated_post"],
        )

        # Print per-distance block in the same style.
        print(f"\n========== Distance = {d} (matched stream, year FE) ==========")
        _print_result(f"Task 1: treated vs control ({outcome_col})", res1)
        _print_result(f"Task 2: pre-retraction treated vs control ({outcome_col})", res2)
        _print_result(f"Task 3: post-retraction treated vs control ({outcome_col})", res3)
        _print_result(f"Task 4: treated-only post vs pre ({outcome_col})", res4)
        msg = "Pretrend leads not implemented in matched streaming mode."
        note = f"  DiD effect is coef on treated_post. {msg}"
        _print_result(f"Task 5A (DiD): distance={d} ({outcome_col})", res5, effect_note=note)
        res5a_by_d[int(d)] = res5



        # Unified 5B accumulation (pooled TWFE with common calendar-year FE across distances)
        if unified_5b:
            if pooled_year_min is None or pooled_mean_y is None or pooled_mean_post is None or pooled_mean_tp is None:
                raise RuntimeError("Unified 5B: pooled year means were not computed")

            if treated_ids_all is None:
                raise RuntimeError("Unified 5B: treated overlap means not computed")

            pos_ids = np.searchsorted(treated_ids_all, ids_s)
            if int(pos_ids.max()) >= int(treated_ids_all.size) or not np.array_equal(treated_ids_all[pos_ids], ids_s):
                raise ValueError(f"Unified 5B: treated ID mapping failed for file {path}")

            cnt_sel = treated_cnt_all[pos_ids].astype(np.float64)
            y_it_all = np.divide(treated_sum_y_all[pos_ids], cnt_sel, out=np.zeros_like(cnt_sel), where=(cnt_sel > 0))
            post_it_all = np.divide(treated_sum_post_all[pos_ids], cnt_sel, out=np.zeros_like(cnt_sel), where=(cnt_sel > 0))
            het_mean_all: dict[int, np.ndarray] = {}
            for dd in het_ds:
                ddi = int(dd)
                het_mean_all[ddi] = np.divide(
                    treated_sum_het_all[ddi][pos_ids],
                    cnt_sel,
                    out=np.zeros_like(cnt_sel),
                    where=(cnt_sel > 0),
                )

            pooled_year_min_i = int(pooled_year_min)

            # Accumulate pooled XtX/Xty by iterating matches once in the control pass (each match once).
            for i in control_rgs:
                rg = pf.read_row_group(i, columns=cols)
                tr = np.asarray(rg.column("treated")).astype(np.int8)
                mask0 = (tr == 0)
                if not mask0.any():
                    continue
                mid = np.asarray(rg.column("corpusid")).astype(np.int64)[mask0]
                idx = _idx_for(mid)
                y_c = _vector_list_col_to_2d_fixed(rg.column(outcome_col), offset=vector_offset, years=T)[mask0].astype(np.float64)
                if transform == "log1p":
                    y_c = np.log1p(y_c)
                finc = np.isfinite(y_c)
                y_cntc3 = finc.sum(axis=1).astype(np.float64)
                y_ic = np.nansum(y_c, axis=1) / np.maximum(y_cntc3, 1.0)
                post_ic = np.divide(
                    (post_f[idx, :] * finc.astype(np.float64)).sum(axis=1),
                    y_cntc3,
                    out=np.zeros_like(y_cntc3),
                    where=(y_cntc3 > 0),
                )

                yt = y_t[idx, :]
                fint = np.isfinite(yt)
                y_it_b = y_it_all[idx]
                post_it_b = post_it_all[idx]

                for j in range(T):
                    yy = year_mat[idx, j] - pooled_year_min_i

                    # treated
                    mt = fint[:, j]
                    if mt.any():
                        y_res = yt[mt, j] - y_it_b[mt] - pooled_mean_y[yy[mt]] + pooled_overall_y
                        x_post = post_f[idx[mt], j] - post_it_b[mt] - pooled_mean_post[yy[mt]] + pooled_overall_post
                        x_tp = post_f[idx[mt], j] - post_it_b[mt] - pooled_mean_tp[yy[mt]] + pooled_overall_tp

                        x_row = np.zeros((int(mt.sum()), k_uni), dtype=np.float64)
                        x_row[:, 0] = x_post
                        x_row[:, 1] = x_tp

                        for dd in het_ds:
                            ddi = int(dd)
                            col = 2 + het_ds.index(ddi)
                            mean_h = pooled_mean_het[ddi][yy[mt]]
                            overall_h = float(pooled_overall_het[ddi])
                            mean_h_p = het_mean_all[ddi][idx[mt]]
                            if int(d) == ddi:
                                x_h = post_f[idx[mt], j] - mean_h_p - mean_h + overall_h
                            else:
                                x_h = 0.0 - mean_h_p - mean_h + overall_h
                            x_row[:, col] = x_h

                        XtX_uni += x_row.T @ x_row
                        Xty_uni += x_row.T @ y_res
                        nobs_uni += int(mt.sum())

                    # control
                    mc = finc[:, j]
                    if mc.any():
                        y_res = y_c[mc, j] - y_ic[mc] - pooled_mean_y[yy[mc]] + pooled_overall_y
                        x_post = post_f[idx[mc], j] - post_ic[mc] - pooled_mean_post[yy[mc]] + pooled_overall_post
                        x_tp = 0.0 - 0.0 - pooled_mean_tp[yy[mc]] + pooled_overall_tp

                        x_row = np.zeros((int(mc.sum()), k_uni), dtype=np.float64)
                        x_row[:, 0] = x_post
                        x_row[:, 1] = x_tp

                        for dd in het_ds:
                            ddi = int(dd)
                            col = 2 + het_ds.index(ddi)
                            mean_h = pooled_mean_het[ddi][yy[mc]]
                            overall_h = float(pooled_overall_het[ddi])
                            x_row[:, col] = 0.0 - 0.0 - mean_h + overall_h

                        XtX_uni += x_row.T @ x_row
                        Xty_uni += x_row.T @ y_res
                        nobs_uni += int(mc.sum())

                # Cluster counts are computed in the meat pass. Here we only accumulate XtX/Xty.

    # End per-distance loop

    if unified_5b:
        # Solve pooled beta
        try:
            beta_uni = np.linalg.solve(XtX_uni, Xty_uni)
        except np.linalg.LinAlgError:
            beta_uni = np.linalg.lstsq(XtX_uni, Xty_uni, rcond=None)[0]

        beta_uni = np.asarray(beta_uni, dtype=np.float64).reshape(-1)
        coef_names_uni = ["post_ret", "treated_post"] + [f"tp_x_dist{dd}" for dd in het_ds]

        print("\nTask 5B unified (matched stream): pooled DiD with distance heterogeneity")
        print(
            "  Model: paper FE + common calendar-year FE (pooled across distances), "
            "cluster=corpusid (treated paper id; allows overlap across distances)"
        )
        print("  Computing cluster-robust SE for pooled model (second pass over data)...")

        # Second pass to compute clustered meat for pooled model.
        scores_by_mid: dict[int, np.ndarray] = {}

        for d, path in inferred:
            cols = ["corpusid", "treated", "publicationdate", "RetractionDate", outcome_col]
            pf = pq.ParquetFile(path)
            treated_rgs, control_rgs = _rowgroup_treated_split(pf)

            # Load treated block
            n_treated = int(sum(pf.metadata.row_group(i).num_rows for i in treated_rgs))
            ids = np.empty(n_treated, dtype=np.int64)
            pub = np.empty(n_treated, dtype=np.int16)
            ret = np.empty(n_treated, dtype=np.int16)
            y_t = np.empty((n_treated, T), dtype=np.float32)
            write = 0

            for i in treated_rgs:
                rg = pf.read_row_group(i, columns=cols)
                tr = np.asarray(rg.column("treated")).astype(np.int8)
                mask = (tr == 1)
                if not mask.any():
                    continue
                mid = np.asarray(rg.column("corpusid")).astype(np.int64)[mask]
                pub_y = np.asarray(pc.year(rg.column("publicationdate")))[mask].astype(np.int16)
                ret_y = np.asarray(pc.year(rg.column("RetractionDate")))[mask].astype(np.int16)
                y_mat = _vector_list_col_to_2d_fixed(rg.column(outcome_col), offset=vector_offset, years=T)[mask].astype(np.float32)
                if transform == "log1p":
                    y_mat = np.log1p(y_mat)

                n = len(mid)
                ids[write : write + n] = mid
                pub[write : write + n] = pub_y
                ret[write : write + n] = ret_y
                y_t[write : write + n, :] = y_mat
                write += n

            if write != n_treated:
                ids = ids[:write]
                pub = pub[:write]
                ret = ret[:write]
                y_t = y_t[:write, :]
                n_treated = write

            order = np.argsort(ids)
            ids_s = ids[order]
            pub_s = pub[order].astype(np.int32)
            ret_s = ret[order].astype(np.int32)
            y_t = y_t[order, :].astype(np.float64)

            year_mat = pub_s[:, None] + ages[None, :].astype(np.int32) - 1
            post = (year_mat >= ret_s[:, None]).astype(np.int8)
            post_f = post.astype(np.float64)

            if pooled_year_min is None or pooled_mean_y is None or pooled_mean_post is None or pooled_mean_tp is None:
                raise RuntimeError("Unified 5B: pooled year means were not computed")
            pooled_year_min_i = int(pooled_year_min)

            y_fin = np.isfinite(y_t)
            # Paper means for treated papers
            y_cnt2 = y_fin.sum(axis=1).astype(np.float64)
            y_it = np.nansum(y_t, axis=1) / np.maximum(y_cnt2, 1.0)
            post_it = np.divide(
                (post_f * y_fin.astype(np.float64)).sum(axis=1),
                y_cnt2,
                out=np.zeros_like(y_cnt2),
                where=(y_cnt2 > 0),
            )

            if treated_ids_all is None:
                raise RuntimeError("Unified 5B: treated overlap means not computed")

            pos_ids = np.searchsorted(treated_ids_all, ids_s)
            if int(pos_ids.max()) >= int(treated_ids_all.size) or not np.array_equal(treated_ids_all[pos_ids], ids_s):
                raise ValueError(f"Unified 5B: treated ID mapping failed for file {path}")
            cnt_sel = treated_cnt_all[pos_ids].astype(np.float64)
            y_it_all = np.divide(treated_sum_y_all[pos_ids], cnt_sel, out=np.zeros_like(cnt_sel), where=(cnt_sel > 0))
            post_it_all = np.divide(treated_sum_post_all[pos_ids], cnt_sel, out=np.zeros_like(cnt_sel), where=(cnt_sel > 0))
            het_mean_all: dict[int, np.ndarray] = {}
            for dd in het_ds:
                ddi = int(dd)
                het_mean_all[ddi] = np.divide(
                    treated_sum_het_all[ddi][pos_ids],
                    cnt_sel,
                    out=np.zeros_like(cnt_sel),
                    where=(cnt_sel > 0),
                )

            beta_vec = np.asarray(beta_uni, dtype=np.float64).reshape(-1)

            for i in control_rgs:
                rg = pf.read_row_group(i, columns=cols)
                tr = np.asarray(rg.column("treated")).astype(np.int8)
                mask0 = (tr == 0)
                if not mask0.any():
                    continue
                mid = np.asarray(rg.column("corpusid")).astype(np.int64)[mask0]
                idx = _idx_for(mid)
                y_c = _vector_list_col_to_2d_fixed(rg.column(outcome_col), offset=vector_offset, years=T)[mask0].astype(np.float64)
                if transform == "log1p":
                    y_c = np.log1p(y_c)
                finc = np.isfinite(y_c)
                y_cntc = np.isfinite(y_c).sum(axis=1).astype(np.float64)
                y_ic = np.nansum(y_c, axis=1) / np.maximum(y_cntc, 1.0)
                post_ic = np.divide(
                    (post_f[idx, :] * finc.astype(np.float64)).sum(axis=1),
                    y_cntc,
                    out=np.zeros_like(y_cntc),
                    where=(y_cntc > 0),
                )
                tp_ic = np.zeros_like(post_ic)

                yt = y_t[idx, :]
                fint = np.isfinite(yt)
                y_it_b = y_it_all[idx]
                post_it_b = post_it_all[idx]

                scores = np.zeros((len(idx), k_uni), dtype=np.float64)
                for j in range(T):
                    yy = year_mat[idx, j] - pooled_year_min_i

                    # treated contribution
                    mt = fint[:, j]
                    if mt.any():
                        y_res = yt[mt, j] - y_it_b[mt] - pooled_mean_y[yy[mt]] + pooled_overall_y
                        x_post = post_f[idx[mt], j] - post_it_b[mt] - pooled_mean_post[yy[mt]] + pooled_overall_post
                        x_tp = post_f[idx[mt], j] - post_it_b[mt] - pooled_mean_tp[yy[mt]] + pooled_overall_tp
                        x_row = np.zeros((int(mt.sum()), k_uni), dtype=np.float64)
                        x_row[:, 0] = x_post
                        x_row[:, 1] = x_tp
                        for dd in het_ds:
                            ddi = int(dd)
                            col = 2 + het_ds.index(ddi)
                            mean_h = pooled_mean_het[ddi][yy[mt]]
                            overall_h = float(pooled_overall_het[ddi])
                            mean_h_p = het_mean_all[ddi][idx[mt]]
                            if int(d) == ddi:
                                x_h = post_f[idx[mt], j] - mean_h_p - mean_h + overall_h
                            else:
                                x_h = 0.0 - mean_h_p - mean_h + overall_h
                            x_row[:, col] = x_h
                        u = y_res - x_row @ beta_vec
                        scores[mt, :] += x_row * u[:, None]

                    # control contribution
                    mc = finc[:, j]
                    if mc.any():
                        y_res = y_c[mc, j] - y_ic[mc] - pooled_mean_y[yy[mc]] + pooled_overall_y
                        x_post = post_f[idx[mc], j] - post_ic[mc] - pooled_mean_post[yy[mc]] + pooled_overall_post
                        x_tp = 0.0 - 0.0 - pooled_mean_tp[yy[mc]] + pooled_overall_tp
                        x_row = np.zeros((int(mc.sum()), k_uni), dtype=np.float64)
                        x_row[:, 0] = x_post
                        x_row[:, 1] = x_tp
                        for dd in het_ds:
                            ddi = int(dd)
                            col = 2 + het_ds.index(ddi)
                            mean_h = pooled_mean_het[ddi][yy[mc]]
                            overall_h = float(pooled_overall_het[ddi])
                            x_row[:, col] = 0.0 - 0.0 - mean_h + overall_h
                        u = y_res - x_row @ beta_vec
                        scores[mc, :] += x_row * u[:, None]

                # Aggregate cluster scores by global corpusid across distances.
                for k, mid_k in enumerate(mid.tolist()):
                    s = scores[k, :]
                    if not np.any(s):
                        continue
                    prev = scores_by_mid.get(int(mid_k))
                    if prev is None:
                        scores_by_mid[int(mid_k)] = s.copy()
                    else:
                        prev += s

        meat_uni = np.zeros((k_uni, k_uni), dtype=np.float64)
        for s in scores_by_mid.values():
            meat_uni += np.outer(s, s)

        n_clusters_uni = int(len(scores_by_mid))

        res_uni = _cluster_robust_kreg(
            XtX=XtX_uni,
            Xty=Xty_uni,
            meat=meat_uni,
            nobs=int(nobs_uni),
            n_clusters=int(n_clusters_uni),
            coef_names=coef_names_uni,
        )

        _print_result(
            "Task 5B unified pooled coefficients (matched stream)",
            res_uni,
            effect_note="  Distance-specific DiD effects are computed as linear combinations of treated_post and tp_x_dist*.",
            cluster_label="corpusid (treated id)",
        )

        # Distance-specific DiD effects implied by the pooled model
        rows: list[tuple[int, float, float, float, float, float, float]] = []
        for dd in run_ds:
            if int(dd) == int(base):
                w = {"treated_post": 1.0}
            else:
                w = {"treated_post": 1.0, f"tp_x_dist{int(dd)}": 1.0}
            est, se, t, p, lo, hi = lincomb(res_uni, w)
            rows.append((int(dd), float(est), float(se), float(t), float(p), float(lo), float(hi)))
        _print_distance_effect_table("Task 5B: distance-specific DiD effects (from unified pooled model)", rows)

        # Heterogeneity test: H0 all tp_x_dist* == 0
        if het_ds:
            het_names = [f"tp_x_dist{int(dd)}" for dd in het_ds]
            stat, df, p = wald_test(res_uni, het_names)
            p_txt = _format_chi2_p(p, False)
            print("\nTask 5B: distance heterogeneity test (pooled model)")
            print(f"  H0: all distance interaction terms = 0 (base distance={base})")
            print(f"  Wald: chi2({df})={stat:.3f}, p={p_txt}")

    if event_study and event_study_csv:
        try:
            pd.DataFrame(es_rows_out).sort_values(["distance", "tau"]).to_csv(event_study_csv, index=False)
            print(f"\n[note] Event-study table saved to: {event_study_csv}")
        except Exception as e:
            print(f"[note] Failed to write event-study CSV: {e}")


def _try_chi2_sf(x: float, df: int) -> float | None:
    """Survival function for chi-square(df) at x.

    Returns None if SciPy is unavailable.
    """
    if df <= 0:
        return None
    stats = _try_scipy_stats()
    if stats is not None:
        try:
            return float(stats.chi2.sf(float(x), df=int(df)))
        except Exception:
            return None

    return None


def _print_distance_effect_table(
    title: str,
    rows: list[tuple[int, float, float, float, float, float, float]],
) -> None:
    """Print a reviewer-friendly distance -> effect table.

    Each row is (distance, est, se, t, p, ci_low, ci_high).
    """
    print(f"\n{title}")
    print("  distance        est        se        t        p                 95% CI")
    for d, est, se, t, p, lo, hi in rows:
        p_txt = _format_p(float(p))
        se_f = float(se)
        if not np.isfinite(se_f):
            se_txt = "NA"
        else:
            se_txt = f"{se_f: .6f}" if abs(se_f) >= 1e-6 else f"{se_f: .6e}"
        est_txt = _fmt_float(float(est), " .6f")
        t_txt = _fmt_float(float(t), " .3f")
        lo_txt = _fmt_float(float(lo), " .6f")
        hi_txt = _fmt_float(float(hi), " .6f")
        print(f"  {int(d):<8} {est_txt}  {se_txt}  {t_txt}  {p_txt:<12}  [{lo_txt}, {hi_txt}]")


def _print_result(
    title: str,
    res: OLSResult,
    effect_note: str | None = None,
    *,
    cluster_label: str = "match_id",
) -> None:
    print(f"\n{title}")
    print(f"nobs={res.nobs:,} | clusters({cluster_label})={res.n_clusters:,}")
    for name, b, se, t, p, lo, hi in zip(
        res.coef_names, res.beta, res.se, res.t, res.p, res.ci_low, res.ci_high
    ):
        p_txt = _format_p(float(p))
        se_f = float(se)
        if not np.isfinite(se_f):
            se_txt = "NA"
        else:
            se_txt = f"{se_f:.6f}" if abs(se_f) >= 1e-6 else f"{se_f:.6e}"

        b_txt = _fmt_float(float(b), " .6f")
        t_txt = _fmt_float(float(t), " .3f")
        lo_txt = _fmt_float(float(lo), " .6f")
        hi_txt = _fmt_float(float(hi), " .6f")
        print(f"  {name:<16} coef={b_txt}  se={se_txt}  t={t_txt}  p={p_txt}  95%CI=[{lo_txt}, {hi_txt}]")
    if effect_note:
        print(effect_note)


def _coef_index(res: OLSResult, name: str) -> int:
    m = {n: i for i, n in enumerate(res.coef_names)}
    if name not in m:
        raise KeyError(f"Coefficient '{name}' not in result. Available: {res.coef_names}")
    return int(m[name])


def wald_test(res: OLSResult, names: list[str]) -> tuple[float, int, float | None]:
    if not names:
        return 0.0, 0, None
    idx = np.array([_coef_index(res, n) for n in names], dtype=int)
    b = res.beta[idx]
    V = res.vcov[np.ix_(idx, idx)]
    try:
        stat = float(b.T @ np.linalg.solve(V, b))
    except np.linalg.LinAlgError:
        stat = float(b.T @ (np.linalg.pinv(V) @ b))
    df = int(len(names))
    p, _used_approx = _chi2_p_value(stat, df)
    return stat, df, p


def lincomb(res: OLSResult, weights: dict[str, float]) -> tuple[float, float, float, float, float, float]:
    w = np.zeros(len(res.beta), dtype=float)
    for name, ww in weights.items():
        w[_coef_index(res, name)] = float(ww)
    est = float(w @ res.beta)
    var = float(w @ res.vcov @ w)
    se = math.sqrt(var) if var >= 0 else float("nan")
    t = est / se if se and np.isfinite(se) else float("nan")
    stats = _require_scipy_stats()
    df = res.n_clusters - 1
    if np.isfinite(t):
        p = float(2.0 * stats.t.sf(abs(t), df=df))
        p = _clip_p01(p)
        tcrit = float(stats.t.ppf(0.975, df=df))
    else:
        p = float("nan")
        tcrit = float("nan")
    lo = est - tcrit * se
    hi = est + tcrit * se
    return est, se, t, p, lo, hi


def fit_fe(
    panel: pd.DataFrame,
    x_cols: list[str],
    fe_cols: list[str],
    cluster_col: str = "match_id",
    *,
    estimator: str = "ols",
) -> OLSResult:
    cols_needed: list[str] = []
    for c in (["y"] + x_cols + fe_cols + [cluster_col]):
        if c not in cols_needed:
            cols_needed.append(c)
    sub = panel[cols_needed].copy()
    if fe_cols:
        sub = sub.dropna(subset=fe_cols + [cluster_col])
    else:
        sub = sub.dropna(subset=[cluster_col])
    for c in ["y"] + x_cols:
        sub = sub[np.isfinite(sub[c].to_numpy(dtype=float))]

    y = sub["y"].to_numpy(dtype=float)
    X = sub[x_cols].to_numpy(dtype=float)

    groups: list[np.ndarray] = []
    n_groups: list[int] = []
    for fe in fe_cols:
        codes, _ = pd.factorize(sub[fe], sort=False)
        groups.append(codes)
        n_groups.append(int(codes.max()) + 1)

    est = estimator.strip().lower()
    if est not in {"ols", "ppml"}:
        raise ValueError("estimator must be 'ols' or 'ppml'")

    if est == "ols":
        if len(groups) == 0:
            y_res, X_res = y, X
        elif len(groups) == 2:
            y_res, X_res = _two_way_demean(y, X, groups[0], n_groups[0], groups[1], n_groups[1])
        else:
            y_res, X_res = _k_way_demean(y, X, groups, n_groups)
        return ols_cluster_robust(y_res, X_res, sub[cluster_col].to_numpy(), coef_names=x_cols)

    # PPML robustness estimator (non-linear; does not support log transform of y)
    return ppml_fe_cluster_robust(
        y=y,
        X=X,
        groups=groups,
        n_groups=n_groups,
        cluster=sub[cluster_col].to_numpy(),
        coef_names=x_cols,
    )


def _time_fe_cols(time_fe: str) -> list[str]:
    t = time_fe.strip().lower()
    if t == "year":
        return ["year"]
    if t == "age":
        return ["age"]
    if t in {"year+age", "age+year"}:
        return ["year", "age"]
    raise ValueError("--time-fe must be one of: year, age, year+age")


def pretrend_leads_test(panel: pd.DataFrame, max_lead: int, time_fe: str) -> tuple[str, float | None]:
    """Event-study leads joint test for DiD (parallel trends diagnostic).

    Uses tau = year - retract_year. Builds lead dummies for tau=-2..-max_lead
    interacted with treated. Reference pre-period is tau=-1.
    """
    if max_lead < 2:
        return "Pretrend test skipped (max_lead < 2)", None
    if panel[["year", "retract_year"]].isna().any().any():
        return "Pretrend test skipped (missing year/retract_year)", None

    tau = panel["year"].astype(int) - panel["retract_year"].astype(int)
    did = panel.copy()
    did["treated_post"] = did["treated"] * did["post_ret"]

    lead_cols: list[str] = []
    for k in range(2, max_lead + 1):
        name = f"lead_m{k}"
        did[name] = (did["treated"] * (tau == -k)).astype(int)
        if did[name].sum() > 0:
            lead_cols.append(name)

    if not lead_cols:
        return "Pretrend test skipped (no lead observations)", None

    fe_cols = ["paper_id"] + _time_fe_cols(time_fe)
    res = fit_fe(
        did,
        x_cols=["post_ret", "treated_post"] + lead_cols,
        fe_cols=fe_cols,
        cluster_col="match_id",
    )
    stat, df, p = wald_test(res, lead_cols)
    p0, used_approx = _chi2_p_value(stat, df)
    return f"Pretrend leads Wald: chi2({df})={stat:.3f}, p={_format_chi2_p(p0, used_approx)}", p0


def run_all_tasks(
    panel: pd.DataFrame,
    outcome_label: str,
    distances: list[int] | None,
    pretrend_leads: int,
    time_fe: str,
    base_distance: int | None = None,
    *,
    estimator: str = "ols",
    event_study: bool = False,
    es_leads: int = 3,
    es_lags: int = 5,
    event_study_csv: str | None = None,
) -> None:
    panel = panel.copy()
    panel["treated"] = panel["treated"].astype(int)
    panel["distance"] = panel["distance"].astype(int)
    panel["post_ret"] = panel["post_ret"].astype(int)

    available = sorted(panel["distance"].dropna().astype(int).unique().tolist())
    use = available if distances is None else [d for d in distances if d in available]
    if not use:
        raise ValueError(f"No requested distances present. Requested={distances}, available={available}")

    print("Data (panel) shape:", panel.shape)
    print("match_id unique:", panel["match_id"].nunique(), "| treated mean:", float(panel["treated"].mean()))
    print("distances available:", available, "| running:", use)
    print("post_ret share:", float(panel["post_ret"].mean()))
    print("outcome:", outcome_label)
    print("time FE spec:", time_fe)

    time_cols = _time_fe_cols(time_fe)
    es_out: list[pd.DataFrame] = []

    # ---- Tasks 1–4: per distance ----
    for d in use:
        sub = panel[panel["distance"] == d]
        print(f"\n========== Distance = {d} ==========")

        # Task 1: overall treated vs control
        res1 = fit_fe(sub, x_cols=["treated"], fe_cols=["match_id"] + time_cols, cluster_col="match_id", estimator=estimator)
        _print_result(f"Task 1: treated vs control ({outcome_label})", res1)

        # Task 2: pre
        pre = sub[sub["post_ret"] == 0]
        if len(pre) > 0:
            res2 = fit_fe(pre, x_cols=["treated"], fe_cols=["match_id"] + time_cols, cluster_col="match_id", estimator=estimator)
            _print_result(f"Task 2: pre-retraction treated vs control ({outcome_label})", res2)
        else:
            print("Task 2 skipped (no pre-retraction observations)")

        # Task 3: post
        post = sub[sub["post_ret"] == 1]
        if len(post) > 0:
            res3 = fit_fe(post, x_cols=["treated"], fe_cols=["match_id"] + time_cols, cluster_col="match_id", estimator=estimator)
            _print_result(f"Task 3: post-retraction treated vs control ({outcome_label})", res3)
        else:
            print("Task 3 skipped (no post-retraction observations)")

        # Task 4: treated-only, post vs pre (within paper)
        treated_only = sub[sub["treated"] == 1]
        if treated_only["post_ret"].nunique() >= 2:
            res4 = fit_fe(treated_only, x_cols=["post_ret"], fe_cols=["paper_id"] + time_cols, cluster_col="match_id", estimator=estimator)
            _print_result(f"Task 4: treated-only post vs pre ({outcome_label})", res4)
        else:
            print("Task 4 skipped (treated-only has no pre/post variation)")

    # ---- Task 5A: DiD per distance ----
    for d in use:
        sub = panel[panel["distance"] == d].copy()
        sub["treated_post"] = sub["treated"] * sub["post_ret"]
        res5a = fit_fe(
            sub,
            x_cols=["post_ret", "treated_post"],
            fe_cols=["paper_id"] + time_cols,
            cluster_col="match_id",
            estimator=estimator,
        )
        msg, _ = pretrend_leads_test(sub, max_lead=pretrend_leads, time_fe=time_fe)
        note = f"  DiD effect is coef on treated_post. {msg}"
        _print_result(f"Task 5A (DiD): distance={d} ({outcome_label})", res5a, effect_note=note)

        if event_study:
            try:
                tab, msg2 = event_study_fe(sub, leads=int(es_leads), lags=int(es_lags), time_fe=time_fe, estimator=estimator)
                if len(tab) > 0:
                    print(f"\nEvent-study note: {msg2}")
                    rows = [
                        (int(r.tau), float(r.est), float(r.se), float(r.t), float(r.p), float(r.ci_low), float(r.ci_high), int(r.n_clusters))
                        for r in tab.itertuples(index=False)
                    ]
                    _print_event_study_table(f"Event-study (TWFE): distance={d} (ref tau=-1)", rows)
                    tab2 = tab.copy()
                    tab2.insert(0, "distance", int(d))
                    es_out.append(tab2)
            except Exception as e:
                print(f"Event-study skipped for distance={d}: {e}")

    # ---- Task 5B: unified DiD with distance heterogeneity ----
    if len(use) == 1:
        d = use[0]
        sub = panel[panel["distance"] == d].copy()
        sub["treated_post"] = sub["treated"] * sub["post_ret"]
        res = fit_fe(sub, x_cols=["post_ret", "treated_post"], fe_cols=["paper_id"] + time_cols, cluster_col="match_id", estimator=estimator)
        msg, _ = pretrend_leads_test(sub, max_lead=pretrend_leads, time_fe=time_fe)
        note = f"  Only distance={d} present, so heterogeneity test is not applicable. {msg}"
        _print_result(
            f"Task 5B (Unified): distance={d} only ({outcome_label})",
            res,
            effect_note=note,
            cluster_label="corpusid (treated id)",
        )
        return

    base = int(base_distance) if base_distance is not None else min(use)
    if base not in use:
        raise ValueError(f"--base-distance={base} not in distances being run: {use}")
    uni = panel[panel["distance"].isin(use)].copy()
    uni["treated_post"] = uni["treated"] * uni["post_ret"]
    het_terms: list[str] = []
    for d in use:
        if d == base:
            continue
        name = f"tp_d{d}"
        uni[name] = uni["treated_post"] * (uni["distance"] == d).astype(int)
        het_terms.append(name)

    res = fit_fe(
        uni,
        x_cols=["post_ret", "treated_post"] + het_terms,
        fe_cols=["paper_id"] + time_cols,
        cluster_col="match_id",
        estimator=estimator,
    )
    msg, _ = pretrend_leads_test(uni, max_lead=pretrend_leads, time_fe=time_fe)
    _print_result(
        f"Task 5B (Unified DiD): base distance={base} ({outcome_label})",
        res,
        effect_note=f"  {msg}",
        cluster_label="corpusid (treated id)",
    )

    rows: list[tuple[int, float, float, float, float, float, float]] = []
    for d in use:
        if d == base:
            est, se, t, p, lo, hi = lincomb(res, {"treated_post": 1.0})
        else:
            est, se, t, p, lo, hi = lincomb(res, {"treated_post": 1.0, f"tp_d{d}": 1.0})
        rows.append((int(d), float(est), float(se), float(t), float(p), float(lo), float(hi)))

    _print_distance_effect_table("Task 5B: distance-specific DiD effects (linear combinations)", rows)

    stat, df, p = wald_test(res, het_terms)
    p0, used_approx = _chi2_p_value(stat, df)
    print(f"\nDistance heterogeneity Wald: chi2({df})={stat:.3f}, p={_format_chi2_p(p0, used_approx)}")

    if event_study and event_study_csv and es_out:
        try:
            pd.concat(es_out, axis=0, ignore_index=True).sort_values(["distance", "tau"]).to_csv(event_study_csv, index=False)
            print(f"\n[note] Event-study table saved to: {event_study_csv}")
        except Exception as e:
            print(f"[note] Failed to write event-study CSV: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Matched-pair retraction analysis (Tasks 1–5)")
    parser.add_argument(
        "--data",
        nargs="+",
        default=["data_1.parquet"],
        help=(
            "One or more parquet files. For per-distance files named data_1.parquet..data_6.parquet, "
            "distance is inferred from filename. For vector_current input, match_id==corpusid (treated-paper id) "
            "is kept as-is (not namespaced) so overlap across distances is accounted for in pooled/unified SE. "
            "For other input formats, IDs may be namespaced across files to avoid accidental collisions when stacking."
        ),
    )
    parser.add_argument(
        "--outcome-col",
        default="harm",
        help="Outcome vector column (vector format) or outcome column (long format). Default: harm.",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=10,
        help="Number of post-publication years to use (default: 10).",
    )
    parser.add_argument(
        "--vector-offset",
        type=int,
        default=1,
        help="Offset into outcome vector for vector format. Default 1 => vector[1]..vector[10] as years 1..10.",
    )
    parser.add_argument(
        "--transform",
        choices=["none", "log1p"],
        default="none",
        help="Optional transform applied to y before regression.",
    )
    parser.add_argument(
        "--distances",
        default="all",
        help="Which distances to run (e.g., '1', '1,2,3', '1-6', or 'all').",
    )
    parser.add_argument(
        "--pretrend-leads",
        type=int,
        default=3,
        help="Max number of pre-period leads for the pretrend joint test (default: 3).",
    )
    parser.add_argument(
        "--estimator",
        choices=["ols", "ppml"],
        default="ols",
        help=(
            "Estimation method. 'ols' is the default (linear FE with cluster-robust SE). "
            "'ppml' runs Poisson pseudo-ML with absorbed FE (robustness; non-stream only recommended)."
        ),
    )
    parser.add_argument(
        "--time-fe",
        choices=["year", "age", "year+age"],
        default="year",
        help=(
            "Time fixed effects specification. "
            "Tasks 1–3 use match_id FE + time FE; Tasks 4–5 use paper_id FE + time FE. "
            "Default: year (calendar-year FE)."
        ),
    )
    parser.add_argument(
        "--base-distance",
        type=int,
        default=None,
        help=(
            "Reference distance for Task 5B unified model interactions. "
            "This only affects coefficient parameterization/labels (not fitted values). "
            "Defaults to the minimum distance being run."
        ),
    )
    parser.add_argument(
        "--event-study",
        action="store_true",
        help=(
            "Run an event-study diagnostic using tau = year - retract_year (ref tau=-1) and print lead/lag coefficients. "
            "In matched streaming year-FE mode, uses within-pair differences dy to stay memory-safe."
        ),
    )
    parser.add_argument(
        "--es-leads",
        type=int,
        default=3,
        help="Event-study pre-period leads window (default: 3). Uses tau=-3..-2 (ref tau=-1).",
    )
    parser.add_argument(
        "--es-lags",
        type=int,
        default=5,
        help="Event-study post-period lags window (default: 5). Uses tau=0..5.",
    )
    parser.add_argument(
        "--event-study-csv",
        default=None,
        help="Optional path to save event-study coefficient table as CSV.",
    )
    parser.add_argument(
        "--sample-matches-per-distance",
        type=int,
        default=0,
        help=(
            "Appendix mode (non-stream): reservoir-sample this many 1:1 matched pairs per distance file "
            "(vector_current parquet only) and run models on the sample. Useful to run unified 5B / TWFE event-study "
            "without loading the full dataset."
        ),
    )
    parser.add_argument(
        "--validate-dy-vs-twfe",
        action="store_true",
        help=(
            "Appendix check: on the sampled data, compare dy-event-study to TWFE event-study for one distance and report "
            "max|difference| and correlation across tau coefficients."
        ),
    )
    parser.add_argument(
        "--validate-distance",
        type=int,
        default=1,
        help="Distance to validate for --validate-dy-vs-twfe (default: 1).",
    )
    parser.add_argument(
        "--validate-seed",
        type=int,
        default=123,
        help="Random seed for appendix sampling/validation (default: 123).",
    )
    parser.add_argument(
        "--stream-vector",
        action="store_true",
        help=(
            "Use low-memory streaming engine for vector_current parquet (large_list outcome like 'harm'). "
            "Use with --time-fe year (or year+age) to run the matched 1:1 engine. "
            "Runs per-distance Tasks 1–5A without building a long panel DataFrame."
        ),
    )
    parser.add_argument(
        "--unified-5b",
        action="store_true",
        help=(
            "Run Task 5B unified model across distances (can be extremely memory-heavy on very large data). "
            "By default, only per-distance Tasks 1–5A are run in a streaming way."
        ),
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Print raw parquet columns/dtypes and head, then exit.",
    )
    args = parser.parse_args()

    if args.estimator == "ppml" and args.transform != "none":
        raise ValueError("PPML does not support --transform log1p; use --transform none")

    # Guidance notes for common identification pitfalls.
    if args.time_fe.strip().lower() in {"year+age", "age+year"} and not args.stream_vector:
        print(
            "[note] --time-fe year+age requested. With paper fixed effects, year and age are typically not separately identified "
            "(year = pub_year + age - 1), so estimates may be numerically unstable or effectively redundant. "
            "For reviewer-facing main specs, prefer --time-fe year (calendar-year shocks) and use --time-fe age as robustness."
        )

    # Lightweight describe using parquet metadata (avoid loading huge files).
    if args.describe:
        try:
            import pyarrow.parquet as pq  # type: ignore

            pf = pq.ParquetFile(args.data[0])
            print("Parquet rows:", pf.metadata.num_rows)
            print("Parquet row_groups:", pf.metadata.num_row_groups)
            print("Columns:", pf.schema.names)
            # Print first 5 rows for key columns only
            cols = pf.schema.names
            key = [c for c in ["corpusid", "treated", "publicationdate", "RetractionDate", args.outcome_col] if c in cols]
            if key:
                head = pq.read_table(args.data[0], columns=key).slice(0, 5).to_pandas()
                print("Head:\n", head.to_string(index=False))
        except Exception as e:
            print("Describe failed:", e)
        return

    distances = _parse_distances(args.distances)

    # ---------------- Appendix sampling mode (non-stream, vector_current only) ----------------
    # This produces reviewer-facing validation / unified regression results without requiring
    # a full-data non-stream load.
    if int(args.sample_matches_per_distance) > 0:
        if args.estimator == "ppml":
            print("[note] Appendix sampling mode with --estimator ppml may be slow; consider --estimator ols for checks.")

        sampled_panels: list[pd.DataFrame] = []
        outcome_label_global: str | None = None
        for p in args.data:
            d_hint = _infer_distance_from_path(p) or 1
            if distances is not None and d_hint not in distances:
                continue

            mids = _sample_match_ids_vector_current_parquet(
                p,
                n_matches=int(args.sample_matches_per_distance),
                seed=int(args.validate_seed) + int(d_hint),
            )
            raw_sub = _load_raw_vector_current_subset(p, match_ids=mids, outcome_col=args.outcome_col)
            if raw_sub.empty:
                continue

            panel_i = _build_panel_vector_current(
                raw_sub,
                outcome_col=args.outcome_col,
                years=args.years,
                vector_offset=args.vector_offset,
                distance_value=int(d_hint),
            )

            if args.transform == "log1p":
                panel_i["y"] = np.log1p(panel_i["y"].astype(float))
                label_i = f"log1p({args.outcome_col})"
            else:
                label_i = args.outcome_col

            outcome_label_global = outcome_label_global or label_i
            sampled_panels.append(panel_i)

        if not sampled_panels:
            raise ValueError("Appendix sampling produced no rows; check --data/--distances")

        panel = pd.concat(sampled_panels, ignore_index=True)
        print(f"[note] Appendix sampling: built panel with nobs={len(panel):,} from per-file samples")

        if args.unified_5b:
            run_all_tasks(
                panel,
                outcome_label=outcome_label_global or args.outcome_col,
                distances=distances,
                pretrend_leads=args.pretrend_leads,
                time_fe=args.time_fe,
                base_distance=args.base_distance,
                estimator=args.estimator,
                event_study=bool(args.event_study),
                es_leads=int(args.es_leads),
                es_lags=int(args.es_lags),
                event_study_csv=args.event_study_csv,
            )
        else:
            available = sorted(panel["distance"].dropna().astype(int).unique().tolist())
            use = available if distances is None else [d for d in distances if d in available]
            for d in use:
                run_tasks_one_distance(
                    panel,
                    outcome_label=outcome_label_global or args.outcome_col,
                    distance_value=int(d),
                    pretrend_leads=args.pretrend_leads,
                    time_fe=args.time_fe,
                    estimator=args.estimator,
                    event_study=bool(args.event_study),
                    es_leads=int(args.es_leads),
                    es_lags=int(args.es_lags),
                )

        if bool(args.validate_dy_vs_twfe):
            d0 = int(args.validate_distance)
            sub = panel[panel["distance"].astype(int) == d0].copy()
            if sub.empty:
                raise ValueError(f"No sampled rows for validate distance={d0}")
            tab_twfe, msg = event_study_fe(
                sub,
                leads=int(args.es_leads),
                lags=int(args.es_lags),
                time_fe=args.time_fe,
                estimator="ols",
            )
            tab_dy = _dy_event_study_on_panel(sub, leads=int(args.es_leads), lags=int(args.es_lags))
            merged = tab_twfe.merge(tab_dy, on="tau", how="inner", suffixes=("_twfe", "_dy"))
            print("\nAppendix validation: dy-event-study vs TWFE event-study (same sampled matches)")
            print(f"  distance={d0} | TWFE leads test: {msg}")
            if len(merged) == 0:
                print("  No overlapping tau rows to compare.")
            else:
                diffs = (merged["est_twfe"] - merged["est_dy"]).to_numpy(dtype=float)
                max_abs = float(np.max(np.abs(diffs)))
                a = merged["est_twfe"].to_numpy(dtype=float)
                b = merged["est_dy"].to_numpy(dtype=float)
                corr = float(np.corrcoef(a, b)[0, 1]) if len(merged) >= 2 else float("nan")
                print(f"  compared tau count={len(merged)} | max|est_twfe-est_dy|={max_abs:.6f} | corr={corr:.4f}")

        return

    # Auto-enable streaming for very large vector_current parquet.
    # IMPORTANT: do not auto-switch to streaming for PPML, because streaming mode is OLS-only and
    # silently changing the estimator would be reviewer-hostile.
    stream = bool(args.stream_vector)
    if (not stream) and (args.estimator == "ols"):
        try:
            import pyarrow.parquet as pq  # type: ignore

            pf0 = pq.ParquetFile(args.data[0])
            cols0 = set(pf0.schema.names)
            if pf0.metadata.num_rows > 2_000_000 and {"corpusid", "treated", "publicationdate", "RetractionDate", args.outcome_col}.issubset(cols0):
                stream = True
        except Exception:
            pass

    if stream:
        if args.estimator == "ppml":
            raise ValueError(
                "PPML is not supported in --stream-vector mode (streaming supports OLS only). "
                "Remove --stream-vector for PPML, or use --estimator ols for streaming."
            )
        # Two streaming engines exist conceptually:
        #   - legacy (age FE): historically used with --time-fe age
        #   - matched 1:1 (calendar-year FE, match clustering): used when --time-fe year/year+age
        # For the provided vector_current data design (corpusid == match_id, duplicated for treated/control),
        # the legacy engine's clustering is not appropriate; we therefore disable it.
        if args.time_fe == "age":
            raise ValueError(
                "--stream-vector with --time-fe age is disabled for correctness. "
                "In vector_current data, corpusid == match_id and is duplicated for treated/control within each pair, "
                "so a legacy paper-level streaming implementation would not cluster correctly. "
                "Use --time-fe year (or year+age) with --stream-vector to run the matched 1:1 streaming engine."
            )

        # Matched 1:1 streaming runner (calendar-year FE). This avoids the non-stream OOM.
        run_stream_vector_matched_1to1(
            args.data,
            outcome_col=args.outcome_col,
            years=args.years,
            vector_offset=args.vector_offset,
            transform=args.transform,
            distances=distances,
            time_fe=args.time_fe,
            pretrend_leads=args.pretrend_leads,
            unified_5b=bool(args.unified_5b),
            base_distance=args.base_distance,
            event_study=bool(args.event_study),
            es_leads=int(args.es_leads),
            es_lags=int(args.es_lags),
            event_study_csv=args.event_study_csv,
        )
        return
    # Memory-friendly default: process each file independently (typical layout is one file per distance).
    # This avoids concatenating all distances into one huge panel.
    outcome_label_global: str | None = None
    for p in args.data:
        forced_d = _infer_distance_from_path(p)
        panel_i, label_i, _fmt_i = load_panel(
            p,
            outcome_col=args.outcome_col,
            years=args.years,
            vector_offset=args.vector_offset,
            transform=args.transform,
            forced_distance=forced_d,
        )
        outcome_label_global = outcome_label_global or label_i

        # Determine which distances to run from this file.
        available = sorted(panel_i["distance"].dropna().astype(int).unique().tolist())
        use = available if distances is None else [d for d in distances if d in available]
        for d in use:
            run_tasks_one_distance(
                panel_i,
                outcome_label=label_i,
                distance_value=int(d),
                pretrend_leads=args.pretrend_leads,
                time_fe=args.time_fe,
                estimator=args.estimator,
                event_study=bool(args.event_study),
                es_leads=int(args.es_leads),
                es_lags=int(args.es_lags),
            )

        # Explicitly drop large objects before next file.
        del panel_i

    if args.unified_5b:
        # Unified 5B requires stacking all distances. This may still be huge.
        panel, outcome_label = load_panels(
            args.data,
            outcome_col=args.outcome_col,
            years=args.years,
            vector_offset=args.vector_offset,
            transform=args.transform,
        )
        run_all_tasks(
            panel,
            outcome_label=outcome_label,
            distances=distances,
            pretrend_leads=args.pretrend_leads,
            time_fe=args.time_fe,
            base_distance=args.base_distance,
            estimator=args.estimator,
            event_study=bool(args.event_study),
            es_leads=int(args.es_leads),
            es_lags=int(args.es_lags),
            event_study_csv=args.event_study_csv,
        )


if __name__ == "__main__":
    main()
