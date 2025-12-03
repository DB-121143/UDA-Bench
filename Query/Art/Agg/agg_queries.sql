-- Query 1: aggregation (Art)
SELECT Style, field, death_city, MIN(teaching) AS min_teaching FROM Art GROUP BY Style;

-- Query 2: aggregation (Art)
SELECT zodiac, birth_date, name, COUNT(name) AS count_name FROM Art GROUP BY zodiac;

-- Query 3: aggregation (Art)
SELECT genre, birth_city, nationality, AVG(age) AS avg_age FROM Art GROUP BY genre;

-- Query 4: aggregation (Art)
SELECT nationality, Theme, art_movement, MAX(age) AS max_age FROM Art GROUP BY nationality;

-- Query 5: aggregation (Art)
SELECT zodiac, birth_date, teaching, COUNT(art_institution) AS count_art_institution FROM Art GROUP BY zodiac;

-- Query 6: aggregation (Art)
SELECT Style, death_city, name, MIN(age) AS min_age FROM Art GROUP BY Style;

-- Query 7: aggregation (Art)
SELECT nationality, birth_continent, name, MAX(age) AS max_age FROM Art GROUP BY nationality;

-- Query 8: aggregation (Art)
SELECT birth_continent, age, nationality, SUM(age) AS sum_age FROM Art GROUP BY birth_continent;

-- Query 9: aggregation (Art)
SELECT nationality, awards, name, COUNT(teaching) AS count_teaching FROM Art GROUP BY nationality;

-- Query 10: aggregation (Art)
SELECT nationality, name, birth_continent, COUNT(birth_city) AS count_birth_city FROM Art GROUP BY nationality;

