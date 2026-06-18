# Exposure research

CrUX / HTTP Archive exposure data for the Script Servedness Score.

## BigQuery preflight

Font exposure is estimated from **HTTP Archive** root-page font requests (Google Fonts CSS and gstatic URLs), mapped to script subsets via the Google Fonts API.

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
export GOOGLE_CLOUD_PROJECT="your-gcp-project"
export GOOGLE_FONTS_API="your-api-key"

python -m exposure_research.bigquery_pull --force
```

Outputs:

- `data/bigquery/big_query_data.csv` — per-font rows with `font_name`, `supported_scripts`, `font_count`
- `data/exposure/exposure_filtered_results.csv` — per-script totals for the heatmap

Optional environment variables:

- `HTTPARCHIVE_CRAWL_DATE` — crawl date (`YYYY-MM-DD`, default: latest)
- `HTTPARCHIVE_CLIENT` — `desktop` or `mobile` (default: `desktop`)
- `SKIP_BIGQUERY=1` — skip pull and use existing CSVs

The exposure pipeline stage (`python pipeline.py --stages exposure`) runs this pull automatically.

## References

- https://har.fyi/guides/getting-started/
- https://har.fyi/guides/minimizing-costs/
- https://github.com/HTTPArchive/almanac.httparchive.org
