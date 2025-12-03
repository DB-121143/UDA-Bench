-- Category 1 : binary_join
-- Query 1: binary_join (player, team)
SELECT team.championships, team.location, player.age, player.olympic_gold_medals FROM player JOIN team ON player.team = team.team_name;

-- Query 2: binary_join (player, team)
SELECT team.founded_year, team.team_name, player.name, player.team FROM player JOIN team ON player.team = team.team_name;

-- Query 3: binary_join (player, team)
SELECT team.ownership, player.name, player.mvp_awards, team.championships FROM player JOIN team ON player.team = team.team_name;

-- Query 4: binary_join (player, team)
SELECT player.nationality, player.age, team.location, team.team_name FROM player JOIN team ON player.team = team.team_name;

-- Query 5: binary_join (team, city)
SELECT city.city_name, city.population, team.founded_year, team.location FROM team JOIN city ON team.location = city.city_name;

-- Query 6: binary_join (team, city)
SELECT city.state_name, city.area, team.founded_year, team.championships FROM team JOIN city ON team.location = city.city_name;

-- Query 7: binary_join (team, manager)
SELECT manager.nba_team, team.location, team.founded_year, manager.name FROM team JOIN manager ON team.ownership = manager.name;

-- Query 8: binary_join (team, manager)
SELECT manager.name, manager.own_year, team.team_name, team.founded_year FROM team JOIN manager ON team.ownership = manager.name;

-- Query 9: binary_join (team, manager)
SELECT manager.nba_team, team.team_name, team.founded_year, manager.nationality FROM team JOIN manager ON team.ownership = manager.name;

-- Query 10: binary_join (team, manager)
SELECT manager.nba_team, team.location, manager.name, team.team_name FROM team JOIN manager ON team.ownership = manager.name;

-- Category 2 : multi_table_join
-- Query 1: multi_table_join (player, team, city)
SELECT player.fiba_world_cup, player.name, team.team_name, team.ownership, city.state_name FROM player JOIN team ON player.team = team.team_name JOIN city ON team.location = city.city_name;

-- Query 2: multi_table_join (player, team, city)
SELECT city.state_name, player.name, player.college, team.founded_year, team.team_name FROM player JOIN team ON player.team = team.team_name JOIN city ON team.location = city.city_name;

-- Query 3: multi_table_join (player, team, city)
SELECT city.population, player.position, city.city_name, team.founded_year, player.name FROM player JOIN team ON player.team = team.team_name JOIN city ON team.location = city.city_name;

-- Query 4: multi_table_join (player, team, city)
SELECT team.championships, city.gdp, player.team, team.team_name, city.city_name FROM player JOIN team ON player.team = team.team_name JOIN city ON team.location = city.city_name;

-- Query 5: multi_table_join (player, team, manager)
SELECT manager.name, team.founded_year, team.location, manager.age, player.nationality FROM player JOIN team ON player.team = team.team_name JOIN manager ON team.ownership = manager.name;

-- Query 6: multi_table_join (player, team, manager)
SELECT player.college, manager.name, player.position, team.championships, player.fiba_world_cup FROM player JOIN team ON player.team = team.team_name JOIN manager ON team.ownership = manager.name;

-- Query 7: multi_table_join (player, team, manager)
SELECT team.championships, manager.age, player.team, player.draft_year, player.age FROM player JOIN team ON player.team = team.team_name JOIN manager ON team.ownership = manager.name;

-- Query 8: multi_table_join (player, team, manager)
SELECT player.nationality, player.olympic_gold_medals, team.location, manager.nba_team, player.draft_pick FROM player JOIN team ON player.team = team.team_name JOIN manager ON team.ownership = manager.name;

-- Query 9: multi_table_join (player, team, city, manager)
SELECT city.state_name, player.olympic_gold_medals, team.team_name, player.college, manager.own_year FROM player JOIN team ON player.team = team.team_name JOIN manager ON team.ownership = manager.name JOIN city ON team.location = city.city_name;

-- Query 10: multi_table_join (player, team, city, manager)
SELECT team.championships, manager.nationality, city.population, player.nationality, player.college FROM player JOIN team ON player.team = team.team_name JOIN manager ON team.ownership = manager.name JOIN city ON team.location = city.city_name;