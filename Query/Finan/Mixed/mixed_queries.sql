-- Query 1: filter1_agg1 (finance)
SELECT auditor, AVG(bussiness_profit) AS avg_bussiness_profit FROM finance WHERE remuneration_policy = 'Performance-based' GROUP BY auditor;

-- Query 2: filter2_agg1 (finance)
SELECT remuneration_policy, MIN(bussiness_profit) AS min_bussiness_profit FROM finance WHERE dividend_per_share > 0.00 AND revenue <= 12857200000 GROUP BY remuneration_policy;

-- Query 3: filter3_agg1 (finance)
SELECT remuneration_policy, AVG(the_highest_ownership_stake) AS avg_the_highest_ownership_stake FROM finance WHERE net_assets <= 249398000 OR total_assets = 1358991000 GROUP BY remuneration_policy;

-- Query 4: filter4_agg1 (finance)
SELECT auditor, MAX(earnings_per_share) AS max_earnings_per_share FROM finance WHERE major_events != 'Litigation' AND remuneration_policy = 'Performance-based' AND total_Debt < '258804000' GROUP BY auditor;

-- Query 5: filter5_agg1 (finance)
SELECT auditor, MIN(earnings_per_share) AS min_earnings_per_share FROM finance WHERE remuneration_policy = 'Performance-based' OR registered_office != '22 Bishopsgate, London, EC2N 4BQ, United Kingdom' OR registered_office != '901 W Walnut Hill Lane, Irving, TX 75038' GROUP BY auditor;

-- Query 6: filter6_agg1 (finance)
SELECT major_equity_changes, SUM(dividend_per_share) AS sum_dividend_per_share FROM finance WHERE (executive_profiles = 'Michael A. Marks' AND net_assets != 1362019954) OR (bussiness_sales != '42984000' AND board_members = 'John Reizenstein') GROUP BY major_equity_changes;

