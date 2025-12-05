-- Query 1: 1 (art)
SELECT death_city, color, birth_date FROM art WHERE field != 'Sculpture';

-- Query 2: 1 (art)
SELECT death_country, death_date, genre FROM art WHERE zodiac = 'Libra';

-- Query 3: 1 (art)
SELECT composition, field, nationality FROM art WHERE nationality != 'Russian';

-- Query 4: 1 (art)
SELECT zodiac, art_movement, object FROM art WHERE birth_country = 'Bulgaria';

-- Query 5: 1 (art)
SELECT theme, tone, birth_date FROM art WHERE birth_date = '1905/4/15';

-- Query 6: 1 (art)
SELECT art_institution, style, marriage FROM art WHERE art_movement = 'Pop art';

-- Query 7: 1 (art)
SELECT marriage, theme, object FROM art WHERE birth_city = 'Boston';

-- Query 8: 1 (art)
SELECT awards, zodiac, object FROM art WHERE death_country = 'Italy';

-- Query 9: 1 (art)
SELECT birth_date, birth_country, tone FROM art WHERE art_institution = '';

-- Query 10: 1 (art)
SELECT genre, style, age FROM art WHERE name != 'Christiaan Karel Appel';

-- Query 11: 2 (art)
SELECT marriage, color, nationality FROM art WHERE genre != 'Expressionist' AND age >= 74;

-- Query 12: 2 (art)
SELECT theme, object, zodiac FROM art WHERE birth_continent = 'Australia' AND art_institution = 'Northwestern University';

-- Query 13: 2 (art)
SELECT color, birth_continent, birth_country FROM art WHERE field != 'Teaching' AND birth_continent = 'Oceania';

-- Query 14: 2 (art)
SELECT genre, birth_date, art_institution FROM art WHERE zodiac != 'Cancer' AND death_city = 'Saint Petersburg';

-- Query 15: 2 (art)
SELECT birth_date, style, theme FROM art WHERE birth_date != '1905/4/27' AND art_movement != 'Cubism';

-- Query 16: 2 (art)
SELECT composition, death_country, birth_date FROM art WHERE birth_date != '1905/3/12' AND death_country = 'Nigeria';

-- Query 17: 2 (art)
SELECT genre, century, birth_continent FROM art WHERE death_date != '1943/1/13' AND birth_city = 'Bucharest';

-- Query 18: 2 (art)
SELECT death_city, marriage, genre FROM art WHERE zodiac = 'Cancer' AND birth_city != 'San Francisco';

-- Query 19: 2 (art)
SELECT color, genre, art_institution FROM art WHERE genre != 'Surrealist' AND teaching < 0;

-- Query 20: 2 (art)
SELECT death_city, birth_continent, style FROM art WHERE birth_date = '1905/3/12' AND marriage = 'Separated';

-- Query 21: 3 (art)
SELECT birth_city, death_date, composition FROM art WHERE death_date != '1943/1/13' OR birth_city != 'Buenos Aires';

-- Query 22: 3 (art)
SELECT color, zodiac, genre FROM art WHERE marriage != 'Cohabiting' OR art_institution != 'Slade School of art';

-- Query 23: 3 (art)
SELECT teaching, death_date, tone FROM art WHERE teaching != 0 OR age != 90;

-- Query 24: 3 (art)
SELECT art_movement, birth_date, style FROM art WHERE art_movement = 'Generación de la Ruptura' OR death_country != 'Brazil';

-- Query 25: 3 (art)
SELECT teaching, color, death_city FROM art WHERE field = 'Drawing' OR death_country != 'France';

-- Query 26: 3 (art)
SELECT composition, awards, century FROM art WHERE century != '20th-21st' OR name != 'George Dawe';

-- Query 27: 3 (art)
SELECT tone, genre, zodiac FROM art WHERE zodiac != 'Cancer' OR death_city = 'Stockholm';

-- Query 28: 3 (art)
SELECT death_country, style, teaching FROM art WHERE art_movement = 'arte Povera' OR name != 'Raoul De Keyser';

-- Query 29: 3 (art)
SELECT genre, death_city, death_country FROM art WHERE zodiac != 'Cancer' OR teaching != 0;

-- Query 30: 3 (art)
SELECT composition, death_country, age FROM art WHERE death_country = 'Ethiopia' OR birth_date = '1908/12/22';

-- Query 31: 4 (art)
SELECT field, color, genre FROM art WHERE field = 'Lithography' AND art_institution != 'Ecole des Beaux-arts' AND death_city != 'New York' AND death_country != 'Russia';

-- Query 32: 4 (art)
SELECT nationality, tone, art_institution FROM art WHERE awards > 1 AND awards >= 0 AND age = 90 AND awards <= 0;

-- Query 33: 4 (art)
SELECT zodiac, composition, teaching FROM art WHERE zodiac = 'Virgo' AND century != '19th-20th' AND teaching >= 0 AND death_country != 'South Africa';

-- Query 34: 4 (art)
SELECT death_city, composition, death_country FROM art WHERE death_city = 'Copenhagen' AND art_institution = 'Royal College of art' AND death_country != 'Hungary' AND death_date = '1969/3/14';

-- Query 35: 4 (art)
SELECT art_movement, style, death_city FROM art WHERE birth_date != '1905/4/25' AND art_institution = 'Claremont Graduate University' AND zodiac = 'Pisces' AND death_city = 'London';

-- Query 36: 4 (art)
SELECT genre, death_city, birth_city FROM art WHERE age <= 83 AND death_city != 'Stuttgart' AND birth_date = '1905/4/4' AND genre = 'Still life';

-- Query 37: 4 (art)
SELECT tone, century, awards FROM art WHERE awards >= 1 AND awards <= 0 AND birth_city != 'Stockholm' AND death_country = 'Switzerland';

-- Query 38: 4 (art)
SELECT death_city, style, art_movement FROM art WHERE name != 'Eduardo Chillida Juantegui' AND age > 90 AND birth_continent != 'Africa' AND birth_country = 'United States';

-- Query 39: 4 (art)
SELECT birth_city, zodiac, tone FROM art WHERE birth_city = 'Addis Ababa' AND death_city != 'Tigre' AND birth_country = 'Switzerland' AND teaching < 0;

-- Query 40: 4 (art)
SELECT marriage, object, awards FROM art WHERE birth_country != 'Spain' AND nationality != 'Canadian' AND teaching >= 0 AND teaching <= 0;

-- Query 41: 5 (art)
SELECT death_city, genre, death_country FROM art WHERE zodiac != 'Cancer' OR birth_date = '1905/4/4' OR art_institution != 'Hunter College' OR awards = 1;

-- Query 42: 5 (art)
SELECT art_institution, nationality, genre FROM art WHERE century = '19th-20th' OR birth_date != '1905/5/12' OR birth_country = 'Chile' OR birth_continent != 'Australia';

-- Query 43: 5 (art)
SELECT tone, birth_city, nationality FROM art WHERE awards = 0 OR birth_city = 'Brussels' OR zodiac != 'Aries' OR age >= 90;

-- Query 44: 5 (art)
SELECT marriage, genre, death_date FROM art WHERE century != '19th' OR name != 'Oswald Achenbach' OR art_institution = 'National Academy of Design' OR teaching > 0;

-- Query 45: 5 (art)
SELECT birth_city, genre, awards FROM art WHERE zodiac = 'Leo' OR marriage != 'Cohabiting' OR name != 'Roy Fox Lichtenstein' OR death_city != 'Stockholm';

-- Query 46: 5 (art)
SELECT object, death_country, age FROM art WHERE death_country != 'United Kingdom' OR field != 'Design' OR nationality != 'Australian' OR awards >= 1;

-- Query 47: 5 (art)
SELECT color, name, age FROM art WHERE field != 'Writer' OR birth_city = 'Paris' OR nationality != 'Indian' OR awards <= 1;

-- Query 48: 5 (art)
SELECT field, teaching, style FROM art WHERE birth_date != '1905/4/15' OR nationality != 'Irish' OR birth_city != 'Warsaw' OR age >= 74;

-- Query 49: 5 (art)
SELECT object, birth_date, death_country FROM art WHERE birth_city = 'Dublin' OR marriage != 'Married' OR teaching != 0 OR birth_city = 'Paris';

-- Query 50: 5 (art)
SELECT name, marriage, composition FROM art WHERE marriage != 'Cohabiting' OR field = 'Painting' OR art_movement = 'art Nouveau' OR field = 'Music';

-- Query 51: 6 (art)
SELECT birth_country, genre, age FROM art WHERE (age >= 61 AND age = 61) OR (death_city != 'Leningrad' AND field != 'Installation art');

-- Query 52: 6 (art)
SELECT composition, age, name FROM art WHERE (age > 90 AND genre != 'Nature') OR (death_city = 'Paris' AND zodiac != 'Aries');

-- Query 53: 6 (art)
SELECT teaching, genre, death_country FROM art WHERE (zodiac = 'Aquarius' AND field = 'Video') OR (teaching != 0 AND birth_city != 'Los Angeles');

-- Query 54: 6 (art)
SELECT death_date, nationality, color FROM art WHERE (marriage = 'Cohabiting' AND birth_continent != 'Australia') OR (nationality = 'Russian' AND death_country != 'Venezuela');

-- Query 55: 6 (art)
SELECT style, genre, zodiac FROM art WHERE (birth_date = '1924/2/9' AND genre != 'Geometric') OR (teaching != 0 AND birth_city = 'Dublin');

-- Query 56: 6 (art)
SELECT age, style, century FROM art WHERE (birth_date != '1905/5/14' AND death_date = '1962/1/16') OR (awards <= 1 AND century != '20th-21st');

-- Query 57: 6 (art)
SELECT color, death_city, teaching FROM art WHERE (field != 'Draughtsmanship' AND awards < 1) OR (name != 'Sir Henry Raeburn' AND name = 'Oscar Agustín Alejandro Schulz Solari');

-- Query 58: 6 (art)
SELECT style, birth_country, theme FROM art WHERE (art_movement != 'Abstract' AND birth_continent != 'South America') OR (death_country != 'Jordan' AND birth_date != '1924/2/9');

-- Query 59: 6 (art)
SELECT field, tone, nationality FROM art WHERE (field != 'Playwriting' AND birth_city = 'New York City') OR (art_institution = 'Ecole des Beaux-arts' AND teaching >= 0);

-- Query 60: 6 (art)
SELECT birth_city, awards, color FROM art WHERE (field = 'Illustration' AND nationality != 'Russian') OR (death_date = '1962/1/16' AND awards != 0);

