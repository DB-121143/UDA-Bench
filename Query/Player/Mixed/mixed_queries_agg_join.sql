-- Query 1: agg1_join1 (player, team)
SELECT player.position, MIN(player.olympic_gold_medals) AS min_player_olympic_gold_medals FROM player JOIN team ON player.team = team.team_name GROUP BY player.position;

-- Query 2: agg1_join2 (player, team, city, manager)
SELECT player.position, MIN(player.olympic_gold_medals) AS min_player_olympic_gold_medals FROM player JOIN team ON player.team = team.team_name JOIN manager ON team.ownership = manager.name JOIN city ON team.location = city.city_name GROUP BY player.position;

