"""Pull font exposure data from HTTP Archive via BigQuery.

Writes:
  - data/bigquery/big_query_data.csv
      font_name, supported_scripts, font_count
  - data/exposure/exposure_filtered_results.csv
      script, count  (canonical names, pilot scripts + Latin)

Requires:
  - google-cloud-bigquery
  - GCP credentials (GOOGLE_APPLICATION_CREDENTIALS or gcloud auth)
  - GOOGLE_FONTS_API for font→subset mapping when CSS subset is absent

Environment:
  HTTPARCHIVE_CRAWL_DATE   YYYY-MM-DD crawl date (default: latest available)
  HTTPARCHIVE_CLIENT       desktop | mobile (default: desktop)
  GOOGLE_CLOUD_PROJECT     GCP project for BigQuery billing
  SKIP_BIGQUERY            set to 1 to skip pull (use existing CSVs)
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd

try:
    from paths import BIGQUERY_DATA_DIR, EXPOSURE_DATA_DIR
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    from paths import BIGQUERY_DATA_DIR, EXPOSURE_DATA_DIR

from exposure_research.script_names import (
    PILOT_SUBSETS,
    normalize_font_key,
    normalize_subset_name,
    to_canonical_script,
)

logger = logging.getLogger(__name__)

SQL_PATH = Path(__file__).resolve().parent / "sql" / "httparchive_font_requests.sql"
BIGQUERY_PATH = BIGQUERY_DATA_DIR / "big_query_data.csv"
EXPOSURE_FILTERED_PATH = EXPOSURE_DATA_DIR / "exposure_filtered_results.csv"


def _fetch_google_font_subsets() -> dict[str, list[str]]:
    """Return font_key → list of Google Fonts subset names."""
    from dotenv import load_dotenv
    import requests

    load_dotenv()
    api_key = os.getenv("GOOGLE_FONTS_API")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_FONTS_API is required to map font families to script subsets."
        )

    url = f"https://www.googleapis.com/webfonts/v1/webfonts?key={api_key}"
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    data = response.json()

    mapping: dict[str, list[str]] = {}
    for item in data.get("items", []):
        family = item.get("family", "")
        subsets = [normalize_subset_name(s) for s in item.get("subsets", [])]
        mapping[normalize_font_key(family)] = subsets
    logger.info("Loaded %d Google Font families for subset mapping", len(mapping))
    return mapping


def _latest_crawl_date(client: str) -> date:
    from google.cloud import bigquery

    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    bq = bigquery.Client(project=project)
    query = """
        SELECT MAX(date) AS latest_date
        FROM `httparchive.crawl.requests`
        WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL 120 DAY)
          AND client = @client
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("client", "STRING", client),
        ]
    )
    rows = list(bq.query(query, job_config=job_config).result())
    latest = rows[0]["latest_date"]
    if latest is None:
        raise RuntimeError("Could not determine latest HTTP Archive crawl date.")
    return latest


def _run_font_request_query(crawl_date: date, client: str) -> pd.DataFrame:
    from google.cloud import bigquery

    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    bq = bigquery.Client(project=project)
    sql = SQL_PATH.read_text(encoding="utf-8")

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("crawl_date", "DATE", crawl_date),
            bigquery.ScalarQueryParameter("client", "STRING", client),
        ]
    )
    logger.info(
        "Running HTTP Archive font query (date=%s, client=%s)...", crawl_date, client
    )
    df = bq.query(sql, job_config=job_config).to_dataframe()
    logger.info("BigQuery returned %d font/subset aggregate rows", len(df))
    return df


def _build_big_query_dataframe(
    raw_df: pd.DataFrame,
    font_subsets: dict[str, list[str]],
) -> pd.DataFrame:
    """Aggregate HTTP Archive counts into one row per font.

    Produces rows with `font_name`, `supported_scripts` (list of canonical
    script display names, e.g. ["Latin","Cyrillic"]) and `font_count`
    (sum of counts for that font across subsets).
    """
    agg: dict[str, dict] = {}

    for _, row in raw_df.iterrows():
        font_key = normalize_font_key(str(row["font_name_raw"]))
        count = int(row["font_count"])
        subset = row.get("subset")
        subset_key = normalize_subset_name(str(subset)) if pd.notna(subset) else None

        # Collect canonical script names for this raw row
        scripts: set[str] = set()
        if subset_key:
            for part in subset_key.split(","):
                normalized = normalize_subset_name(part.strip())
                canonical = to_canonical_script(normalized)
                if canonical:
                    scripts.add(canonical)

        # Fall back to Google Fonts mapping when subset info is absent
        if not scripts:
            for s in font_subsets.get(font_key, []):
                canonical = to_canonical_script(normalize_subset_name(s))
                if canonical:
                    scripts.add(canonical)

        if not scripts:
            continue

        if font_key not in agg:
            agg[font_key] = {"count": 0, "scripts": set()}
        agg[font_key]["count"] += count
        agg[font_key]["scripts"].update(scripts)

    if not agg:
        raise RuntimeError(
            "No font rows matched pilot script subsets. "
            "Check HTTP Archive results and Google Fonts API mapping."
        )

    records: list[dict] = []
    for font_name, meta in agg.items():
        scripts_list = sorted(meta["scripts"])
        records.append(
            {
                "font_name": font_name,
                "supported_scripts": scripts_list,
                "font_count": meta["count"],
            }
        )

    df = pd.DataFrame(records)
    df = df.sort_values("font_count", ascending=False).reset_index(drop=True)
    return df

    if not records:
        raise RuntimeError(
            "No font rows matched pilot script subsets. "
            "Check HTTP Archive results and Google Fonts API mapping."
        )

    df = pd.DataFrame(records)
    # Merge duplicate font+script combinations from separate subset rows
    exploded = df.explode("supported_scripts")
    grouped = (
        exploded.groupby(["font_name", "supported_scripts"], as_index=False)["font_count"]
        .sum()
    )
    grouped["supported_scripts"] = grouped["supported_scripts"].apply(
        lambda s: [normalize_subset_name(s)]
    )
    return grouped


def _build_exposure_filtered(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per canonical script name for heatmap / clustering inputs."""
    exploded = df.copy()
    exploded["script"] = exploded["supported_scripts"].apply(
        lambda scripts: normalize_subset_name(scripts[0])
    )
    totals = (
        exploded.groupby("script", as_index=False)["font_count"]
        .sum()
        .rename(columns={"font_count": "count"})
    )
    totals["canonical"] = totals["script"].map(to_canonical_script)
    totals = totals[totals["canonical"].notna()]
    summary = (
        totals.groupby("canonical", as_index=False)["count"]
        .sum()
        .rename(columns={"canonical": "script"})
        .sort_values("count", ascending=False)
    )
    return summary


def pull_exposure_data(
    crawl_date: date | None = None,
    client: str | None = None,
    force: bool = False,
) -> tuple[Path, Path]:
    """Run BigQuery pull and write CSV outputs."""
    if os.getenv("SKIP_BIGQUERY") == "1":
        logger.warning("SKIP_BIGQUERY=1 — using existing BigQuery CSVs")
        if not BIGQUERY_PATH.exists():
            raise FileNotFoundError(f"Missing {BIGQUERY_PATH}")
        return BIGQUERY_PATH, EXPOSURE_FILTERED_PATH

    BIGQUERY_DATA_DIR.mkdir(parents=True, exist_ok=True)
    EXPOSURE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if BIGQUERY_PATH.exists() and EXPOSURE_FILTERED_PATH.exists() and not force:
        logger.info("BigQuery outputs already exist (use --force to rerun)")
        return BIGQUERY_PATH, EXPOSURE_FILTERED_PATH

    client = client or os.getenv("HTTPARCHIVE_CLIENT", "desktop")
    crawl_date = crawl_date or _parse_env_date() or _latest_crawl_date(client)

    font_subsets = _fetch_google_font_subsets()
    raw_df = _run_font_request_query(crawl_date, client)
    bq_df = _build_big_query_dataframe(raw_df, font_subsets)

    # Serialize list column for CSV using the legacy bracketed format: [A,B,C]
    bq_df["supported_scripts"] = bq_df["supported_scripts"].apply(
        lambda scripts: "[" + ",".join(scripts) + "]"
    )
    bq_df.to_csv(BIGQUERY_PATH, index=False)
    logger.info("Wrote %s", BIGQUERY_PATH)

    # Re-parse for exposure summary (lists stored as bracketed strings)
    def _parse_scripts_field(s: str) -> list[str]:
        s = str(s).strip()
        inner = s.strip("[]").strip()
        if not inner:
            return []
        return [part.strip() for part in inner.split(",") if part.strip()]

    bq_df["supported_scripts"] = bq_df["supported_scripts"].apply(_parse_scripts_field)
    filtered = _build_exposure_filtered(bq_df)
    filtered.to_csv(EXPOSURE_FILTERED_PATH, index=False)
    logger.info("Wrote %s", EXPOSURE_FILTERED_PATH)

    return BIGQUERY_PATH, EXPOSURE_FILTERED_PATH


def _parse_env_date() -> date | None:
    raw = os.getenv("HTTPARCHIVE_CRAWL_DATE")
    if not raw:
        return None
    return date.fromisoformat(raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull HTTP Archive font exposure data from BigQuery"
    )
    parser.add_argument(
        "--crawl-date",
        type=str,
        default=None,
        help="HTTP Archive crawl date (YYYY-MM-DD). Default: latest available.",
    )
    parser.add_argument(
        "--client",
        choices=["desktop", "mobile"],
        default=None,
        help="HTTP Archive client (default: desktop).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun even if output files already exist.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = parse_args()
    crawl_date = date.fromisoformat(args.crawl_date) if args.crawl_date else None
    pull_exposure_data(crawl_date=crawl_date, client=args.client, force=args.force)


if __name__ == "__main__":
    main()
