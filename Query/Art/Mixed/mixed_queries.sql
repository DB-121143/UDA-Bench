-- Query 1: filter1_agg1 (art)
SELECT field, AVG(age) AS avg_age FROM art WHERE century = '19th-20th' GROUP BY field;

-- Query 2: filter2_agg1 (art)
SELECT nationality, COUNT(birth_continent) AS count_birth_continent FROM art WHERE nationality = 'Japanese' AND birth_continent != 'Australia' GROUP BY nationality;

-- Query 3: filter3_agg1 (art)
SELECT genre, AVG(age) AS avg_age FROM art WHERE birth_country = 'British India' OR teaching != 0 GROUP BY genre;

-- Query 4: filter4_agg1 (art)
SELECT image_genre, MAX(awards) AS max_awards FROM art WHERE art_institution != 'National Academy of Design' AND name != 'Emma Amos' AND birth_date = '1905/5/11' GROUP BY image_genre;

-- Query 5: filter5_agg1 (art)
SELECT color, MAX(awards) AS max_awards FROM art WHERE awards > 0 OR birth_country = 'Kingdom of Hungary' OR birth_country = 'Italy' GROUP BY color;

-- Query 6: filter6_agg1 (art)
SELECT nationality, COUNT(nationality) AS count_nationality FROM art WHERE (nationality = 'German-American' AND teaching = 1) OR (birth_continent != 'Australia' AND birth_date = '1905/5/15') GROUP BY nationality;

