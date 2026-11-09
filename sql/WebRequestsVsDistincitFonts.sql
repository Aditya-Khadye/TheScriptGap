-- Query used in the generation of "Web requests / Distinct fonts" aka Table 2
-- HTTP Archive Data on Font Requests
-- The following is based on codepoints and not expanded data from multiple sources
--     ie. the subset of scripts that the website supports by default
-- # of Requests for a script
-- # of Distinct Font Names per Script


SELECT
  script AS Script,
  COUNT(*) AS Font_Requests,
  COUNT(DISTINCT font_name) AS Distinct_Font_Count
FROM `scriptgap.font_requests.20260301`,
UNNEST(scripts) AS script
WHERE script IN ('Latin', 'Arabic','Devanagari','Bengali','Tamil','Telugu','Han','Katakana','Cyrillic') -- comment this out to remove the filter
GROUP BY script
ORDER BY Font_Requests DESC