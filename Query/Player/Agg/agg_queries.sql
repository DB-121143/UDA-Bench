-- Query 1: aggregation (player)
SELECT position, MIN(olympic_gold_medals) AS min_olympic_gold_medals FROM player GROUP BY position;

-- Query 2: aggregation (player)
SELECT nationality, COUNT(*) AS count_all FROM player GROUP BY nationality;

-- Query 3: aggregation (player)
SELECT position, AVG(age) AS avg_age FROM player GROUP BY position;

-- Query 4: aggregation (player)
SELECT nationality, MAX(age) AS max_age FROM player GROUP BY nationality;

-- Query 5: aggregation (player)
SELECT nationality, COUNT(*) AS count_all FROM player GROUP BY nationality;

-- Query 6: aggregation (player)
SELECT nationality, MIN(age) AS min_age FROM player GROUP BY nationality;

-- Query 7: aggregation (player)
SELECT nationality, AVG(age) AS avg_age FROM player GROUP BY nationality;

-- Query 8: aggregation (manager)
SELECT nationality, MIN(age) AS min_age FROM manager GROUP BY nationality;

-- Query 9: aggregation (manager)
SELECT nationality, COUNT(*) AS count_all FROM manager GROUP BY nationality;

-- Query 10: aggregation (manager)
SELECT nationality, AVG(age) AS avg_age FROM manager GROUP BY nationality;

