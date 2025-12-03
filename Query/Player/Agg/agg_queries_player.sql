-- Query 1: aggregation (player)
SELECT position, MIN(olympic_gold_medals) AS min_olympic_gold_medals FROM player GROUP BY position;

-- Query 2: aggregation (player)
SELECT nationality, COUNT(name) AS count_name FROM player GROUP BY nationality;

-- Query 3: aggregation (player)
SELECT position, AVG(age) AS avg_age FROM player GROUP BY position;

-- Query 4: aggregation (player)
SELECT nationality, MAX(age) AS max_age FROM player GROUP BY nationality;

-- Query 5: aggregation (player)
SELECT nationality, COUNT(college) AS count_college FROM player GROUP BY nationality;

-- Query 6: aggregation (player)
SELECT position, MIN(age) AS min_age FROM player GROUP BY position;

-- Query 7: aggregation (player)
SELECT nationality, MAX(age) AS max_age FROM player GROUP BY nationality;

