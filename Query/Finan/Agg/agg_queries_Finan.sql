-- Query 1: aggregation (Finan)
SELECT remuneration_policy, MIN(the_highest_ownership_stake) AS min_the_highest_ownership_stake FROM Finan GROUP BY remuneration_policy;

-- Query 2: aggregation (Finan)
SELECT major_equity_changes, COUNT(company_name) AS count_company_name FROM Finan GROUP BY major_equity_changes;

-- Query 3: aggregation (Finan)
SELECT remuneration_policy, AVG(executive_profiles) AS avg_executive_profiles FROM Finan GROUP BY remuneration_policy;

-- Query 4: aggregation (Finan)
SELECT major_equity_changes, MAX(executive_profiles) AS max_executive_profiles FROM Finan GROUP BY major_equity_changes;

-- Query 5: aggregation (Finan)
SELECT major_equity_changes, COUNT(major_equity_changes) AS count_major_equity_changes FROM Finan GROUP BY major_equity_changes;

-- Query 6: aggregation (Finan)
SELECT remuneration_policy, MIN(executive_profiles) AS min_executive_profiles FROM Finan GROUP BY remuneration_policy;

-- Query 7: aggregation (Finan)
SELECT major_equity_changes, MAX(executive_profiles) AS max_executive_profiles FROM Finan GROUP BY major_equity_changes;

-- Query 8: aggregation (Finan)
SELECT major_equity_changes, SUM(executive_profiles) AS sum_executive_profiles FROM Finan GROUP BY major_equity_changes;

-- Query 9: aggregation (Finan)
SELECT major_equity_changes, COUNT(auditor) AS count_auditor FROM Finan GROUP BY major_equity_changes;

-- Query 10: aggregation (Finan)
SELECT major_equity_changes, COUNT(total_assets) AS count_total_assets FROM Finan GROUP BY major_equity_changes;

