-- Query 1: filter1_agg1_join1 (player, team)
SELECT player.nationality, MIN(player.olympic_gold_medals) AS min_player_olympic_gold_medals FROM player JOIN team ON player.team = team.team_name WHERE player.fiba_world_cup > 0 GROUP BY player.nationality;

-- Query 2: filter2_agg1_join2 (player, team, city, manager)
SELECT manager.nationality, MAX(player.mvp_awards) AS max_player_mvp_awards FROM player JOIN team ON player.team = team.team_name JOIN manager ON team.ownership = manager.name JOIN city ON team.location = city.city_name WHERE city.population < '1603797' OR team.location != 'Minneapolis' GROUP BY manager.nationality;

-- Query 3: filter3_agg1_join1 (player, team)
SELECT player.nationality, AVG(player.mvp_awards) AS avg_player_mvp_awards FROM player JOIN team ON player.team = team.team_name WHERE player.age <= 66 GROUP BY player.nationality;

-- Query 4: filter4_agg1_join2 (player, team, city, manager)
SELECT manager.nationality, MAX(player.nba_championships) AS max_player_nba_championships FROM player JOIN team ON player.team = team.team_name JOIN manager ON team.ownership = manager.name JOIN city ON team.location = city.city_name WHERE city.population = '1603797' AND city.gdp != '518.5' AND player.birth_date = '1990/8/17' GROUP BY manager.nationality;

-- Query 5: filter5_agg1_join1 (player, team)
SELECT player.position, AVG(player.fiba_world_cup) AS avg_player_fiba_world_cup FROM player JOIN team ON player.team = team.team_name WHERE player.nba_championships < 0 OR player.age >= 47 OR player.nationality = 'Greek-American  ' GROUP BY player.position;

-- Query 6: filter6_agg1_join2 (player, team, city, manager)
SELECT player.position, AVG(player.age) AS avg_player_age FROM player JOIN team ON player.team = team.team_name JOIN manager ON team.ownership = manager.name JOIN city ON team.location = city.city_name WHERE (player.nba_championships <= 2 AND player.mvp_awards < 1) OR (player.olympic_gold_medals != 0 AND player.nationality = 'American-born naturalized Azerbaijani  ') GROUP BY player.position;

-- Query 7: filter1_agg1_join1 (player, team)
SELECT player.position, AVG(player.fiba_world_cup) AS avg_player_fiba_world_cup FROM player JOIN team ON player.team = team.team_name WHERE player.olympic_gold_medals = 0 GROUP BY player.position;

-- Query 8: filter2_agg1_join2 (player, team, city, manager)
SELECT player.position, AVG(player.olympic_gold_medals) AS avg_player_olympic_gold_medals FROM player JOIN team ON player.team = team.team_name JOIN manager ON team.ownership = manager.name JOIN city ON team.location = city.city_name WHERE manager.nationality = 'Israeli-American' AND team.founded_year > 1989 GROUP BY player.position;

-- Query 9: filter3_agg1_join1 (player, team)
SELECT player.nationality, COUNT(player.nationality) AS count_player_nationality FROM player JOIN team ON player.team = team.team_name WHERE player.nationality = 'Cameroonian-American' OR player.birth_date = '1972/3/6' GROUP BY player.nationality;

-- Query 10: filter4_agg1_join2 (player, team, city, manager)
SELECT player.nationality, SUM(team.championships) AS sum_team_championships FROM player JOIN team ON player.team = team.team_name JOIN manager ON team.ownership = manager.name JOIN city ON team.location = city.city_name WHERE player.draft_year <= 1990 AND player.position = 'Backcourt' AND player.draft_pick <= 35 GROUP BY player.nationality;

-- Query 11: filter5_agg1_join1 (player, team)
SELECT player.nationality, SUM(player.mvp_awards) AS sum_player_mvp_awards FROM player JOIN team ON player.team = team.team_name WHERE player.team != 'Milwaukee Bucks' OR player.nba_championships > 2 OR team.founded_year != 1949 GROUP BY player.nationality;

-- Query 12: filter6_agg1_join2 (player, team, city, manager)
SELECT player.position, MIN(team.championships) AS min_team_championships FROM player JOIN team ON player.team = team.team_name JOIN manager ON team.ownership = manager.name JOIN city ON team.location = city.city_name WHERE (team.location = 'Brooklyn' AND player.nba_championships < 0) OR (player.team != 'Milwaukee Hawks  ' AND player.draft_pick != 35) GROUP BY player.position;

