-- Query 1: filter1_agg1 (Finan)
SELECT remuneration_policy, AVG(executive_profiles) AS avg_executive_profiles FROM Finan WHERE net_assets != 8699800000 GROUP BY remuneration_policy;

-- Query 2: filter2_agg1 (Finan)
SELECT remuneration_policy, AVG(the_highest_ownership_stake) AS avg_the_highest_ownership_stake FROM Finan WHERE total_Debt = '104280399' AND net_profit_or_loss != '160800000' GROUP BY remuneration_policy;

-- Query 3: filter3_agg1 (Finan)
SELECT major_equity_changes, MAX(executive_profiles) AS max_executive_profiles FROM Finan WHERE major_equity_changes >= 'Yes' OR dividend_per_share = 1.12 GROUP BY major_equity_changes;

-- Query 4: filter4_agg1 (Finan)
SELECT major_equity_changes, MAX(executive_profiles) AS max_executive_profiles FROM Finan WHERE bussiness_profit = '1925000000' AND executive_profiles = '4 (Ganesh Pattabiraman' AND bussiness_profit = '28900000' GROUP BY major_equity_changes;

-- Query 5: filter5_agg1 (Finan)
SELECT remuneration_policy, AVG(executive_profiles) AS avg_executive_profiles FROM Finan WHERE cash_reserves != 1987600000 OR auditor = 'PricewaterhouseCoopers LLP' OR total_assets = 136955488 GROUP BY remuneration_policy;

-- Query 6: filter6_agg1 (Finan)
SELECT major_equity_changes, SUM(executive_profiles) AS sum_executive_profiles FROM Finan WHERE (board_members = 'Yanbin Wang' AND business_segments_num >= 4) OR (revenue = 12857200000 AND executive_profiles != 'Bethany Greenwood') GROUP BY major_equity_changes;

