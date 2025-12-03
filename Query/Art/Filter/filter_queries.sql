-- Query 1: 1 (Art)
SELECT death_city, Color, birth_date FROM Art WHERE field != 'Sculpture';

-- Query 2: 1 (Art)
SELECT death_country, death_date, Genre FROM Art WHERE zodiac = 'Libra';

-- Query 3: 1 (Art)
SELECT Composition, field, nationality FROM Art WHERE nationality != 'Russian';

-- Query 4: 1 (Art)
SELECT zodiac, art_movement, Object FROM Art WHERE birth_country = 'Bulgaria';

-- Query 5: 1 (Art)
SELECT Theme, Tone, birth_date FROM Art WHERE birth_date = '1905/4/15';

-- Query 6: 1 (Art)
SELECT art_institution, Style, marriage FROM Art WHERE art_movement = 'Pop Art';

-- Query 7: 1 (Art)
SELECT marriage, Theme, Object FROM Art WHERE birth_city = 'Boston';

-- Query 8: 1 (Art)
SELECT awards, zodiac, Object FROM Art WHERE death_country = 'Italy';

-- Query 9: 1 (Art)
SELECT birth_date, birth_country, Tone FROM Art WHERE art_institution = '';

-- Query 10: 1 (Art)
SELECT genre, Style, age FROM Art WHERE name != 'Christiaan Karel Appel';

-- Query 11: 2 (Art)
SELECT marriage, Color, nationality FROM Art WHERE genre != 'Expressionist' AND age >= 74;

-- Query 12: 2 (Art)
SELECT Theme, Object, zodiac FROM Art WHERE birth_continent = 'Australia' AND art_institution = 'Northwestern University';

-- Query 13: 2 (Art)
SELECT Color, birth_continent, birth_country FROM Art WHERE field != 'Teaching' AND birth_continent = 'Oceania';

-- Query 14: 2 (Art)
SELECT Genre, birth_date, art_institution FROM Art WHERE zodiac != 'Cancer' AND death_city = 'Saint Petersburg';

-- Query 15: 2 (Art)
SELECT birth_date, Style, Theme FROM Art WHERE birth_date != '1905/4/27' AND art_movement != 'Cubism';

-- Query 16: 2 (Art)
SELECT Composition, death_country, birth_date FROM Art WHERE birth_date != '1905/3/12' AND death_country = 'Nigeria';

-- Query 17: 2 (Art)
SELECT Genre, century, birth_continent FROM Art WHERE death_date != '1943/1/13' AND birth_city = 'Bucharest';

-- Query 18: 2 (Art)
SELECT death_city, marriage, Genre FROM Art WHERE zodiac = 'Cancer' AND birth_city != 'San Francisco';

-- Query 19: 2 (Art)
SELECT Color, genre, art_institution FROM Art WHERE genre != 'Surrealist' AND teaching < 0;

-- Query 20: 2 (Art)
SELECT death_city, birth_continent, Style FROM Art WHERE birth_date = '1905/3/12' AND marriage = 'Separated';

-- Query 21: 3 (Art)
SELECT birth_city, death_date, Composition FROM Art WHERE death_date != '1943/1/13' OR birth_city != 'Buenos Aires';

-- Query 22: 3 (Art)
SELECT Color, zodiac, genre FROM Art WHERE marriage != 'Cohabiting' OR art_institution != 'Slade School of Art';

-- Query 23: 3 (Art)
SELECT teaching, death_date, Tone FROM Art WHERE teaching != 0 OR age != 90;

-- Query 24: 3 (Art)
SELECT art_movement, birth_date, Style FROM Art WHERE art_movement = 'Generación de la Ruptura' OR death_country != 'Brazil';

-- Query 25: 3 (Art)
SELECT teaching, Color, death_city FROM Art WHERE field = 'Drawing' OR death_country != 'France';

-- Query 26: 3 (Art)
SELECT Composition, awards, century FROM Art WHERE century != '20th-21st' OR name != 'George Dawe';

-- Query 27: 3 (Art)
SELECT Tone, genre, zodiac FROM Art WHERE zodiac != 'Cancer' OR death_city = 'Stockholm';

-- Query 28: 3 (Art)
SELECT death_country, Style, teaching FROM Art WHERE art_movement = 'Arte Povera' OR name != 'Raoul De Keyser';

-- Query 29: 3 (Art)
SELECT Genre, death_city, death_country FROM Art WHERE zodiac != 'Cancer' OR teaching != 0;

-- Query 30: 3 (Art)
SELECT Composition, death_country, age FROM Art WHERE death_country = 'Ethiopia' OR birth_date = '1908/12/22';

-- Query 31: 4 (Art)
SELECT field, Color, genre FROM Art WHERE field = 'Lithography' AND art_institution != 'Ecole des Beaux-Arts' AND death_city != 'New York' AND death_country != 'Russia';

-- Query 32: 4 (Art)
SELECT nationality, Tone, art_institution FROM Art WHERE awards > 1 AND awards >= 0 AND age = 90 AND awards <= 0;

-- Query 33: 4 (Art)
SELECT zodiac, Composition, teaching FROM Art WHERE zodiac = 'Virgo' AND century != '19th-20th' AND teaching >= 0 AND death_country != 'South Africa';

-- Query 34: 4 (Art)
SELECT death_city, Composition, death_country FROM Art WHERE death_city = 'Copenhagen' AND art_institution = 'Royal College of Art' AND death_country != 'Hungary' AND death_date = '1969/3/14';

-- Query 35: 4 (Art)
SELECT art_movement, Style, death_city FROM Art WHERE birth_date != '1905/4/25' AND art_institution = 'Claremont Graduate University' AND zodiac = 'Pisces' AND death_city = 'London';

-- Query 36: 4 (Art)
SELECT Genre, death_city, birth_city FROM Art WHERE age <= 83 AND death_city != 'Stuttgart' AND birth_date = '1905/4/4' AND genre = 'Still life';

-- Query 37: 4 (Art)
SELECT Tone, century, awards FROM Art WHERE awards >= 1 AND awards <= 0 AND birth_city != 'Stockholm' AND death_country = 'Switzerland';

-- Query 38: 4 (Art)
SELECT death_city, Style, art_movement FROM Art WHERE name != 'Eduardo Chillida Juantegui' AND age > 90 AND birth_continent != 'Africa' AND birth_country = 'United States';

-- Query 39: 4 (Art)
SELECT birth_city, zodiac, Tone FROM Art WHERE birth_city = 'Addis Ababa' AND death_city != 'Tigre' AND birth_country = 'Switzerland' AND teaching < 0;

-- Query 40: 4 (Art)
SELECT marriage, Object, awards FROM Art WHERE birth_country != 'Spain' AND nationality != 'Canadian' AND teaching >= 0 AND teaching <= 0;

-- Query 41: 5 (Art)
SELECT death_city, Genre, death_country FROM Art WHERE zodiac != 'Cancer' OR birth_date = '1905/4/4' OR art_institution != 'Hunter College' OR awards = 1;

-- Query 42: 5 (Art)
SELECT art_institution, nationality, Genre FROM Art WHERE century = '19th-20th' OR birth_date != '1905/5/12' OR birth_country = 'Chile' OR birth_continent != 'Australia';

-- Query 43: 5 (Art)
SELECT Tone, birth_city, nationality FROM Art WHERE awards = 0 OR birth_city = 'Brussels' OR zodiac != 'Aries' OR age >= 90;

-- Query 44: 5 (Art)
SELECT marriage, Genre, death_date FROM Art WHERE century != '19th' OR name != 'Oswald Achenbach' OR art_institution = 'National Academy of Design' OR teaching > 0;

-- Query 45: 5 (Art)
SELECT birth_city, Genre, awards FROM Art WHERE zodiac = 'Leo' OR marriage != 'Cohabiting' OR name != 'Roy Fox Lichtenstein' OR death_city != 'Stockholm';

-- Query 46: 5 (Art)
SELECT Object, death_country, age FROM Art WHERE death_country != 'United Kingdom' OR field != 'Design' OR nationality != 'Australian' OR awards >= 1;

-- Query 47: 5 (Art)
SELECT Color, name, age FROM Art WHERE field != 'Writer' OR birth_city = 'Paris' OR nationality != 'Indian' OR awards <= 1;

-- Query 48: 5 (Art)
SELECT field, teaching, Style FROM Art WHERE birth_date != '1905/4/15' OR nationality != 'Irish' OR birth_city != 'Warsaw' OR age >= 74;

-- Query 49: 5 (Art)
SELECT Object, birth_date, death_country FROM Art WHERE birth_city = 'Dublin' OR marriage != 'Married' OR teaching != 0 OR birth_city = 'Paris';

-- Query 50: 5 (Art)
SELECT name, marriage, Composition FROM Art WHERE marriage != 'Cohabiting' OR field = 'Painting' OR art_movement = 'Art Nouveau' OR field = 'Music';

-- Query 51: 6 (Art)
SELECT birth_country, Genre, age FROM Art WHERE (age >= 61 AND age = 61) OR (death_city != 'Leningrad' AND field != 'Installation Art');

-- Query 52: 6 (Art)
SELECT Composition, age, name FROM Art WHERE (age > 90 AND genre != 'Nature') OR (death_city = 'Paris' AND zodiac != 'Aries');

-- Query 53: 6 (Art)
SELECT teaching, Genre, death_country FROM Art WHERE (zodiac = 'Aquarius' AND field = 'Video') OR (teaching != 0 AND birth_city != 'Los Angeles');

-- Query 54: 6 (Art)
SELECT death_date, nationality, Color FROM Art WHERE (marriage = 'Cohabiting' AND birth_continent != 'Australia') OR (nationality = 'Russian' AND death_country != 'Venezuela');

-- Query 55: 6 (Art)
SELECT Style, genre, zodiac FROM Art WHERE (birth_date = '1924/2/9' AND genre != 'Geometric') OR (teaching != 0 AND birth_city = 'Dublin');

-- Query 56: 6 (Art)
SELECT age, Style, century FROM Art WHERE (birth_date != '1905/5/14' AND death_date = '1962/1/16') OR (awards <= 1 AND century != '20th-21st');

-- Query 57: 6 (Art)
SELECT Color, death_city, teaching FROM Art WHERE (field != 'Draughtsmanship' AND awards < 1) OR (name != 'Sir Henry Raeburn' AND name = 'Oscar Agustín Alejandro Schulz Solari');

-- Query 58: 6 (Art)
SELECT Style, birth_country, Theme FROM Art WHERE (art_movement != 'Abstract' AND birth_continent != 'South America') OR (death_country != 'Jordan' AND birth_date != '1924/2/9');

-- Query 59: 6 (Art)
SELECT field, Tone, nationality FROM Art WHERE (field != 'Playwriting' AND birth_city = 'New York City') OR (art_institution = 'Ecole des Beaux-Arts' AND teaching >= 0);

-- Query 60: 6 (Art)
SELECT birth_city, awards, Color FROM Art WHERE (field = 'Illustration' AND nationality != 'Russian') OR (death_date = '1962/1/16' AND awards != 0);

