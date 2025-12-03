-- Query 1: aggregation (manager)
SELECT nationality, MIN(age) AS min_age FROM manager GROUP BY nationality;

-- Query 2: aggregation (manager)
SELECT nationality, COUNT(name) AS count_name FROM manager GROUP BY nationality;

-- Query 3: aggregation (manager)
SELECT nationality, AVG(age) AS avg_age FROM manager GROUP BY nationality;

