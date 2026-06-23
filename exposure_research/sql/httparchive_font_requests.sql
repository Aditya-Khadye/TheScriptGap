-- HTTP Archive: Google Fonts request counts by family and optional CSS subset.
-- Processes ~tens of GB depending on crawl month; run via exposure_research/bigquery_pull.py.
--
-- Parameters (passed by the Python client):
--   crawl_date  DATE   HTTP Archive monthly crawl (YYYY-MM-01)
--   client      STRING 'desktop' or 'mobile'

WITH raw AS (
  SELECT
    CASE
      WHEN url LIKE '%fonts.googleapis.com/css%' THEN
        LOWER(
          TRIM(
            REPLACE(
              REPLACE(
                REGEXP_EXTRACT(url, r'family=([^|&:]+)'),  -- stops at first |, &, or :
                '+', ' '
              ),
              '%20', ' '
            )
          )
        )
      WHEN url LIKE '%fonts.gstatic.com/s/%' THEN
        LOWER(TRIM(REPLACE(REGEXP_EXTRACT(url, r'fonts\.gstatic\.com/s/([^/]+)/'), '-', ' ')))
      ELSE NULL
    END AS font_name_raw,
    CASE
      WHEN url LIKE '%fonts.googleapis.com/css%' AND REGEXP_CONTAINS(url, r'subset=') THEN
        LOWER(REGEXP_EXTRACT(url, r'subset=([^&]+)'))
      ELSE NULL
    END AS subset
  FROM `httparchive.crawl.requests`
  WHERE date = @crawl_date
    AND client = @client
    AND is_root_page
    AND (
      url LIKE '%fonts.googleapis.com%'
      OR url LIKE '%fonts.gstatic.com%'
    )
)
SELECT
  font_name_raw,
  subset,
  COUNT(*) AS font_count
FROM raw
WHERE font_name_raw IS NOT NULL
  AND font_name_raw != ''
  AND REGEXP_CONTAINS(font_name_raw, r'^[a-z0-9 -]+$')  -- filter malformed entries
GROUP BY font_name_raw, subset
ORDER BY font_count DESC
