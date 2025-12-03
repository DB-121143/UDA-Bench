-- Query 1: 1 (city)
SELECT state_name, area, city_name FROM city WHERE area = 375.78;

-- Query 2: 2 (city)
SELECT state_name, area, city_name FROM city WHERE area != 1314.80 AND population = '887642';

-- Query 3: 3 (city)
SELECT state_name, city_name, population FROM city WHERE state_name != 'Indiana' OR population != '372,624';

-- Query 4: 4 (city)
SELECT gdp, area, state_name FROM city WHERE area = 976.15 AND gdp != '473,000' AND area = 375.78 AND population = '808988';

-- Query 5: 5 (city)
SELECT gdp, state_name, population FROM city WHERE state_name != 'Ontario' OR city_name = 'Minneapolis' OR gdp != '102,000,000,000' OR gdp != ' ';

-- Query 6: 6 (city)
SELECT population, state_name, city_name FROM city WHERE (state_name = 'Minnesota' AND population = '887642') OR (area != 976.15 AND area != 976.15);

