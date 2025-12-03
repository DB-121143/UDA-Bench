-- Query 1: 1 (manager)
SELECT age, nba_team, name FROM manager WHERE nba_team != 'Cleveland Cavaliers';

-- Query 2: 2 (manager)
SELECT age, nba_team, name FROM manager WHERE nba_team != 'Golden State Warriors' AND age >= 66;

-- Query 3: 3 (manager)
SELECT age, name, nationality FROM manager WHERE age < 63 OR nationality != 'Israeli-American';

-- Query 4: 4 (manager)
SELECT own_year, nba_team, age FROM manager WHERE nba_team = 'Miami Heat' AND own_year != 2008 AND nationality != 'American  ' AND nationality != 'American  ';

-- Query 5: 5 (manager)
SELECT own_year, age, nationality FROM manager WHERE age <= 76 OR name = 'Joseph Chung-Hsin Tsai' OR own_year != 2012 OR own_year = 2017;

-- Query 6: 6 (manager)
SELECT nationality, age, name FROM manager WHERE (age >= 63 AND age = 63) OR (nba_team != 'Indiana Pacers' AND nba_team != 'New Orleans Pelicans');

