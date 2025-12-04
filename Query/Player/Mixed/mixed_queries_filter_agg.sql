-- Query 1: filter1_agg1 (manager)
SELECT nationality, MIN(age) AS min_age FROM manager WHERE nba_team != 'Cleveland Cavaliers' GROUP BY nationality;

-- Query 2: filter2_agg1 (manager)
SELECT nationality, SUM(age) AS sum_age FROM manager WHERE age != 76 AND nba_team != 'Houston Rockets' GROUP BY nationality;

-- Query 3: filter3_agg1 (player)
SELECT nationality, COUNT(age) AS count_age FROM player WHERE name = 'Antonius Cleveland' OR nba_championships >= 0 GROUP BY nationality;

-- Query 4: filter4_agg1 (player)
SELECT nationality, MAX(mvp_awards) AS max_mvp_awards FROM data WHERE draft_year > 2012 OR nba_championships > 0 GROUP BY nationality;

-- Query 5: filter5_agg1 (player)
SELECT nationality, AVG(fiba_world_cup) AS avg_fiba_world_cup FROM player WHERE team = 'New York Knicks  ' OR nba_championships <= 0 OR fiba_world_cup < 0 GROUP BY nationality;

-- Query 6: filter6_agg1 (player)
SELECT position, MIN(fiba_world_cup) AS min_fiba_world_cup FROM player WHERE (fiba_world_cup < 0 AND mvp_awards <= 0) OR (age < 91 AND fiba_world_cup >= 0) GROUP BY position;

