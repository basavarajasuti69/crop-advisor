# Outbreak Dashboard — Athena SQL queries for crop disease analytics
#
# Run these in the Athena console (workgroup: crop-advisor-dev-analytics)
# against the glue database: crop_advisor_analytics

-- 1. Disease distribution by crop (last 30 days)
SELECT
  crop,
  predicted_label AS disease,
  COUNT(*) AS case_count,
  ROUND(AVG(confidence) * 100, 1) AS avg_confidence_pct
FROM cases
WHERE received_at >= from_iso8601_timestamp(now() - interval '30' day)
GROUP BY crop, predicted_label
ORDER BY case_count DESC;

-- 2. Outbreak hotspots by location (high-risk diseases)
SELECT
  location,
  crop,
  predicted_label AS disease,
  COUNT(*) AS case_count,
  COUNT(*) FILTER (WHERE escalate = true) AS escalated_count
FROM cases
WHERE received_at >= from_iso8601_timestamp(now() - interval '7' day)
  AND predicted_label <> 'healthy'
GROUP BY location, crop, predicted_label
HAVING COUNT(*) >= 3
ORDER BY escalated_count DESC, case_count DESC;

-- 3. Daily diagnosis trend
SELECT
  date_parse(substr(received_at, 1, 10), '%Y-%m-%d') AS day,
  COUNT(*) AS total_cases,
  COUNT(*) FILTER (WHERE predicted_label <> 'healthy') AS disease_cases,
  ROUND(COUNT(*) FILTER (WHERE predicted_label <> 'healthy') * 100.0 / COUNT(*), 1) AS disease_rate_pct
FROM cases
GROUP BY date_parse(substr(received_at, 1, 10), '%Y-%m-%d')
ORDER BY day DESC
LIMIT 30;

-- 4. Confidence distribution (model quality monitoring)
SELECT
  CASE
    WHEN confidence >= 0.9 THEN 'high (>=0.9)'
    WHEN confidence >= 0.7 THEN 'medium (0.7-0.9)'
    WHEN confidence >= 0.5 THEN 'low (0.5-0.7)'
    ELSE 'very_low (<0.5)'
  END AS confidence_bucket,
  COUNT(*) AS case_count
FROM cases
GROUP BY 1
ORDER BY case_count DESC;

-- 5. Language usage
SELECT
  language,
  COUNT(*) AS case_count
FROM cases
GROUP BY language
ORDER BY case_count DESC;

-- 6. Create table DDL (run once after Glue crawler populates)
-- CREATE EXTERNAL TABLE IF NOT EXISTS cases (
--   case_id string,
--   crop string,
--   language string,
--   location string,
--   predicted_label string,
--   predicted_score double,
--   confidence double,
--   escalate boolean,
--   escalate_reason string,
--   answer string,
--   received_at string
-- )
-- STORED AS PARQUET
-- LOCATION 's3://<analytics-bucket>/cases/';
