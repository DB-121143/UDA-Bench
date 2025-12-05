-- Query 1: aggregation (art)
SELECT teaching, MAX(awards) AS max_awards FROM art GROUP BY teaching;

-- Query 2: aggregation (art)
SELECT zodiac, COUNT(name) AS count_name FROM art GROUP BY zodiac;

-- Query 3: aggregation (art)
SELECT genre, AVG(age) AS avg_age FROM art GROUP BY genre;

-- Query 4: aggregation (art)
SELECT color, AVG(age) AS avg_age FROM art GROUP BY color;

-- Query 5: aggregation (art)
SELECT zodiac, COUNT(art_institution) AS count_art_institution FROM art GROUP BY zodiac;

-- Query 6: aggregation (art)
SELECT teaching, MIN(age) AS min_age FROM art GROUP BY teaching;

-- Query 7: aggregation (art)
SELECT color, MAX(age) AS max_age FROM art GROUP BY color;

-- Query 8: aggregation (art)
SELECT birth_continent, SUM(age) AS sum_age FROM art GROUP BY birth_continent;

