-- Query 1: aggregation (Finan)
SELECT remuneration_policy, MIN(business_segments_num) AS min_business_segments_num FROM Finan GROUP BY remuneration_policy;

-- Query 2: aggregation (Finan)
SELECT auditor, COUNT(company_name) AS count_company_name FROM Finan GROUP BY auditor;

-- Query 3: aggregation (Finan)
SELECT auditor, AVG(business_segments_num) AS avg_business_segments_num FROM Finan GROUP BY auditor;

-- Query 4: aggregation (Finan)
SELECT auditor, MAX(revenue) AS max_revenue FROM Finan GROUP BY auditor;

-- Query 5: aggregation (Finan)
SELECT auditor, COUNT(major_equity_changes) AS count_major_equity_changes FROM Finan GROUP BY auditor;

-- Query 6: aggregation (Finan)
SELECT remuneration_policy, MIN(revenue) AS min_revenue FROM Finan GROUP BY remuneration_policy;

-- Query 7: aggregation (Finan)
SELECT auditor, MAX(revenue) AS max_revenue FROM Finan GROUP BY auditor;

-- Query 8: aggregation (Finan)
SELECT major_equity_changes, SUM(revenue) AS sum_revenue FROM Finan GROUP BY major_equity_changes;

