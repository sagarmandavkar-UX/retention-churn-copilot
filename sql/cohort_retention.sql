-- DuckDB/Postgres-style cohort retention using explicit account-month grain.
WITH activity AS (
    SELECT account_id, DATE_TRUNC('month', event_at) AS activity_month
    FROM product_events
    GROUP BY 1, 2
), cohorts AS (
    SELECT account_id, MIN(activity_month) AS cohort_month
    FROM activity
    GROUP BY 1
), retained AS (
    SELECT
        c.cohort_month,
        DATE_DIFF('month', c.cohort_month, a.activity_month) AS month_number,
        COUNT(DISTINCT a.account_id) AS retained_accounts
    FROM activity a
    JOIN cohorts c USING (account_id)
    GROUP BY 1, 2
), sizes AS (
    SELECT cohort_month, retained_accounts AS cohort_size
    FROM retained
    WHERE month_number = 0
)
SELECT
    r.cohort_month,
    r.month_number,
    r.retained_accounts,
    s.cohort_size,
    r.retained_accounts * 1.0 / s.cohort_size AS retention_rate
FROM retained r
JOIN sizes s USING (cohort_month)
ORDER BY 1, 2;
