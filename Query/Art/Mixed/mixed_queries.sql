-- Query 1: filter1_agg1 (Art)
SELECT nationality, COUNT(birth_continent) AS count_birth_continent FROM Art WHERE nationality = 'Spanish' GROUP BY nationality;

-- Query 2: filter2_agg1 (Art)
SELECT field, MAX(birth_country) AS max_birth_country FROM Art WHERE awards = 1 AND death_country = 'Georgia' GROUP BY field;

-- Query 3: filter3_agg1 (Art)
SELECT field, SUM(teaching) AS sum_teaching FROM Art WHERE century = '20th-21st' OR teaching != 1 GROUP BY field;

-- Query 4: filter4_agg1 (Art)
SELECT marriage, AVG(death_country) AS avg_death_country FROM Art WHERE birth_continent = 'Oceania' AND name != 'George Frederic Watts' AND death_date = '1905/5/13' GROUP BY marriage;

-- Query 5: filter5_agg1 (Art)
SELECT marriage, AVG(awards) AS avg_awards FROM Art WHERE death_country = 'Australia' OR field != 'Visual Art' OR art_movement = 'Art Nouveau' GROUP BY marriage;

-- Query 6: filter6_agg1 (Art)
SELECT genre, AVG(death_country) AS avg_death_country FROM Art WHERE (birth_city = 'Tbilisi' AND birth_city != 'Montevideo') OR (teaching > 0 AND death_country = 'Bangladesh') GROUP BY genre;

