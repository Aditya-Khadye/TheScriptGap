-- Script Gap Font Requests from HTTP Archive 20260301 crawl
-- Has Font Requests key values extracted from the HTTP Archive to make queries more efficient.
-- Scripts derived from codepoints
-- Example Return:
-- font_name,font_count,scripts
-- roboto,2853622,"[Hebrew,PUA,Greek,Inherited,Latin,Georgian,Cyrillic]"

WITH 
flattened AS (
  SELECT font_name, script
  FROM `scriptgap.font_requests.20260301`,
  UNNEST(scripts) AS script
  WHERE font_name IS NOT NULL
),

-- Aggregates the scripts that a font supports
fonts_n_scripts AS (
  SELECT 
    font_name,
    ARRAY_AGG(DISTINCT script) AS scripts
  FROM flattened
  GROUP BY font_name
),

-- Gets font count for the font name
-- Count font name excludes nulls. If you want nulls use *
font_count AS (SELECT font_name, COUNT(font_name) AS font_count 
FROM `scriptgap.font_requests.20260301`
WHERE font_name IS NOT NULL
GROUP BY font_name
ORDER BY font_count DESC)

SELECT *
FROM font_count
LEFT JOIN fonts_n_scripts USING (font_name)
ORDER BY font_count DESC;