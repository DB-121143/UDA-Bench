-- Query 1: aggregation (Finan_finance)
SELECT remuneration_policy, dividend_per_share, earnings_per_share, MIN(the_highest_ownership_stake) AS min_the_highest_ownership_stake FROM Finan_finance GROUP BY remuneration_policy;

-- Query 2: aggregation (Finan_finance)
SELECT major_equity_changes, principal_activities, business_segments_num, COUNT(company_name) AS count_company_name FROM Finan_finance GROUP BY major_equity_changes;

-- Query 3: aggregation (Finan_finance)
SELECT remuneration_policy, total_assets, auditor, AVG(executive_profiles) AS avg_executive_profiles FROM Finan_finance GROUP BY remuneration_policy;

-- Query 4: aggregation (Finan_finance)
SELECT major_equity_changes, bussiness_profit, registered_office, MAX(executive_profiles) AS max_executive_profiles FROM Finan_finance GROUP BY major_equity_changes;

-- Query 5: aggregation (Finan_finance)
SELECT major_equity_changes, auditor, principal_activities, COUNT(major_equity_changes) AS count_major_equity_changes FROM Finan_finance GROUP BY major_equity_changes;

-- Query 6: aggregation (Finan_finance)
SELECT remuneration_policy, earnings_per_share, company_name, MIN(executive_profiles) AS min_executive_profiles FROM Finan_finance GROUP BY remuneration_policy;

-- Query 7: aggregation (Finan_finance)
SELECT major_equity_changes, bussiness_cost, company_name, MAX(executive_profiles) AS max_executive_profiles FROM Finan_finance GROUP BY major_equity_changes;

-- Query 8: aggregation (Finan_finance)
SELECT major_equity_changes, executive_profiles, registered_office, SUM(executive_profiles) AS sum_executive_profiles FROM Finan_finance GROUP BY major_equity_changes;

-- Query 9: aggregation (Finan_finance)
SELECT major_equity_changes, auditor, company_name, COUNT(auditor) AS count_auditor FROM Finan_finance GROUP BY major_equity_changes;

-- Query 10: aggregation (Finan_finance)
SELECT major_equity_changes, company_name, total_assets, COUNT(total_assets) AS count_total_assets FROM Finan_finance GROUP BY major_equity_changes;

