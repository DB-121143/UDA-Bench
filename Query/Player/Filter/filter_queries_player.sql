-- Query 1: 1 (player)
SELECT mvp_awards, draft_pick, fiba_world_cup FROM player WHERE fiba_world_cup <= 0;

-- Query 2: 1 (player)
SELECT position, nationality, age FROM player WHERE age < 91;

-- Query 3: 1 (player)
SELECT mvp_awards, draft_pick, name FROM player WHERE mvp_awards >= 1;

-- Query 4: 1 (player)
SELECT age, birth_date, team FROM player WHERE team = 'Phoenix Suns';

-- Query 5: 1 (player)
SELECT fiba_world_cup, nba_championships, birth_date FROM player WHERE nba_championships = 0;

-- Query 6: 1 (player)
SELECT college, birth_date, draft_year FROM player WHERE birth_date = '1973/11/25';

-- Query 7: 2 (player)
SELECT fiba_world_cup, draft_year, name FROM player WHERE draft_year <= 2017 AND fiba_world_cup > 0;

-- Query 8: 2 (player)
SELECT position, nba_championships, fiba_world_cup FROM player WHERE fiba_world_cup > 0 AND nationality != 'French  ';

-- Query 9: 2 (player)
SELECT draft_pick, nba_championships, team FROM player WHERE draft_pick < 17 AND draft_pick >= 5;

-- Query 10: 2 (player)
SELECT age, birth_date, college FROM player WHERE age > 91 AND mvp_awards > 0;

-- Query 11: 2 (player)
SELECT fiba_world_cup, birth_date, nba_championships FROM player WHERE birth_date != '1959/6/10' AND birth_date != '1964/2/15';

-- Query 12: 2 (player)
SELECT olympic_gold_medals, position, birth_date FROM player WHERE olympic_gold_medals < 1 AND draft_pick = 17;

-- Query 13: 3 (player)
SELECT team, nationality, mvp_awards FROM player WHERE mvp_awards < 1 OR birth_date = '1995/10/2';

-- Query 14: 3 (player)
SELECT draft_year, age, fiba_world_cup FROM player WHERE draft_year <= 2017 OR olympic_gold_medals >= 0;

-- Query 15: 3 (player)
SELECT fiba_world_cup, nationality, college FROM player WHERE college = 'UCLA  ' OR birth_date != '1971/12/3';

-- Query 16: 3 (player)
SELECT olympic_gold_medals, fiba_world_cup, birth_date FROM player WHERE birth_date = '1994/4/25' OR position != 'Frontcourt';

-- Query 17: 3 (player)
SELECT college, draft_pick, fiba_world_cup FROM player WHERE draft_pick <= 5 OR college = 'Wake Forest University';

-- Query 18: 3 (player)
SELECT olympic_gold_medals, nba_championships, age FROM player WHERE olympic_gold_medals > 1 OR position != 'Frontcourt';

-- Query 19: 4 (player)
SELECT fiba_world_cup, draft_pick, draft_year FROM player WHERE draft_pick >= 17 AND age >= 47 AND mvp_awards <= 0 AND mvp_awards < 1;

-- Query 20: 4 (player)
SELECT name, nba_championships, college FROM player WHERE nba_championships > 2 AND olympic_gold_medals >= 0 AND olympic_gold_medals != 1 AND name != 'Toby Kimball  ';

-- Query 21: 4 (player)
SELECT age, olympic_gold_medals, mvp_awards FROM player WHERE olympic_gold_medals >= 1 AND college = 'UCLA  ' AND draft_year > 2012 AND fiba_world_cup >= 0;

-- Query 22: 4 (player)
SELECT draft_pick, olympic_gold_medals, position FROM player WHERE olympic_gold_medals > 0 AND birth_date != '1943/12/23' AND nba_championships < 2 AND birth_date = '1992/12/17';

-- Query 23: 4 (player)
SELECT fiba_world_cup, birth_date, draft_pick FROM player WHERE birth_date != '1950/1/29' AND college = 'University of Florida' AND age != 47 AND name = 'Walter Berry ';

-- Query 24: 4 (player)
SELECT nationality, draft_pick, team FROM player WHERE nationality != 'Croatian  ' AND olympic_gold_medals < 0 AND olympic_gold_medals != 0 AND mvp_awards = 0;

-- Query 25: 5 (player)
SELECT draft_pick, age, position FROM player WHERE age <= 66 OR birth_date = '1997/8/7' OR college != 'Duke University' OR nba_championships = 2;

-- Query 26: 5 (player)
SELECT olympic_gold_medals, college, age FROM player WHERE age > 47 OR name != 'Fran Curran Francis Hugh Curran Sr.' OR name = 'Dewayne "D. J." White, Jr.' OR fiba_world_cup = 1;

-- Query 27: 5 (player)
SELECT nba_championships, name, olympic_gold_medals FROM player WHERE nba_championships = 0 OR team != 'Miami Heat  ' OR nationality = ' ' OR team != 'Philadelphia 76ers  ';

-- Query 28: 5 (player)
SELECT draft_year, age, nationality FROM player WHERE age <= 91 OR team = 'Guaros de Lara' OR mvp_awards = 0 OR olympic_gold_medals > 1;

-- Query 29: 5 (player)
SELECT olympic_gold_medals, age, team FROM player WHERE age >= 47 OR team != 'San Antonio Spurs' OR nba_championships <= 0 OR nba_championships < 0;

-- Query 30: 5 (player)
SELECT position, nationality, olympic_gold_medals FROM player WHERE position != 'Frontcourt' OR draft_pick > 5 OR olympic_gold_medals < 0 OR mvp_awards = 0;

-- Query 31: 6 (player)
SELECT team, nationality, fiba_world_cup FROM player WHERE (nationality = 'American-Venezuelan  ' AND position = 'Frontcourt') OR (draft_pick >= 17 AND draft_pick >= 17);

-- Query 32: 6 (player)
SELECT nationality, olympic_gold_medals, fiba_world_cup FROM player WHERE (fiba_world_cup != 0 AND name = 'Erick Strickland  ') OR (birth_date != '1973/11/25' AND nba_championships >= 0);

-- Query 33: 6 (player)
SELECT college, age, position FROM player WHERE (age < 47 AND olympic_gold_medals <= 1) OR (team = 'Milwaukee Hawks  ' AND college = 'University of Florida  ');

-- Query 34: 6 (player)
SELECT nationality, name, draft_year FROM player WHERE (draft_year != 2017 AND nationality != 'Croatian  ') OR (name = 'Donta Hall  ' AND nba_championships != 0);

-- Query 35: 6 (player)
SELECT birth_date, olympic_gold_medals, mvp_awards FROM player WHERE (birth_date = '1973/11/25' AND olympic_gold_medals != 0) OR (team = 'Miami Heat' AND nationality = 'Canadian');

-- Query 36: 6 (player)
SELECT nationality, birth_date, age FROM player WHERE (birth_date != '1994/6/6' AND nationality = 'Dutch  ') OR (nba_championships <= 2 AND age > 91);

