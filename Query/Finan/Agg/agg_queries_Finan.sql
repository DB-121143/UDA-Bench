-- Query 1: aggregation (finance)
SELECT remuneration_policy, MIN(business_segments_num) AS min_business_segments_num FROM finance GROUP BY remuneration_policy;

-- Query 2: aggregation (finance)
SELECT auditor, COUNT(company_name) AS count_company_name FROM finance GROUP BY auditor;

-- Query 3: aggregation (finance)
SELECT auditor, AVG(business_segments_num) AS avg_business_segments_num FROM finance GROUP BY auditor;

-- Query 4: aggregation (finance)
SELECT auditor, MAX(revenue) AS max_revenue FROM finance GROUP BY auditor;

-- Query 5: aggregation (finance)
SELECT auditor, COUNT(major_equity_changes) AS count_major_equity_changes FROM finance GROUP BY auditor;

-- Query 6: aggregation (finance)
SELECT remuneration_policy, MIN(revenue) AS min_revenue FROM finance GROUP BY remuneration_policy;

-- Query 7: aggregation (finance)
SELECT auditor, MAX(revenue) AS max_revenue FROM finance GROUP BY auditor;

-- Query 8: aggregation (finance)
SELECT major_equity_changes, SUM(revenue) AS sum_revenue FROM finance GROUP BY major_equity_changes;

