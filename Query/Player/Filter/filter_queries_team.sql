-- Query 1: 1 (team)
SELECT founded_year, ownership, team_name FROM team WHERE ownership != '  ';

-- Query 2: 1 (team)
SELECT ownership, championships, founded_year FROM team WHERE founded_year < 1989;

-- Query 3: 2 (team)
SELECT founded_year, ownership, team_name FROM team WHERE ownership != 'Harris Blitzer Sports & Entertainment (HBSE)  ' AND founded_year >= 1967;

-- Query 4: 2 (team)
SELECT championships, founded_year, location FROM team WHERE location = 'Brooklyn' AND location = 'Memphis';

-- Query 5: 3 (team)
SELECT founded_year, team_name, location FROM team WHERE founded_year < 1949 OR location != 'Oklahoma City';

-- Query 6: 3 (team)
SELECT ownership, founded_year, championships FROM team WHERE ownership != 'Professional Basketball Club LLC, a group of Oklahoma City investors led by Clay Bennett  ' OR ownership != 'Jerry Buss (from 1979)';

-- Query 7: 4 (team)
SELECT championships, ownership, founded_year FROM team WHERE ownership = 'James L. Dolan' AND ownership = 'Glen Taylor' AND location = 'Houston' AND location != 'Brooklyn';

-- Query 8: 4 (team)
SELECT team_name, championships, location FROM team WHERE team_name = 'Golden State Warriors' AND founded_year >= 1989 AND ownership != 'Joseph Tsai' AND location != 'Charlotte';

-- Query 9: 5 (team)
SELECT championships, founded_year, location FROM team WHERE founded_year <= 1978 OR team_name = 'Dallas Mavericks' OR location != 'Minneapolis' OR team_name != 'Miami Heat';

-- Query 10: 5 (team)
SELECT team_name, founded_year, ownership FROM team WHERE founded_year > 1967 OR team_name != 'Charlotte Hornets' OR team_name = 'Detroit Pistons' OR founded_year >= 1989;

-- Query 11: 6 (team)
SELECT location, founded_year, team_name FROM team WHERE (founded_year >= 1949 AND founded_year = 1949) OR (ownership != 'Paul Allen' AND ownership != 'Glen Taylor');

-- Query 12: 6 (team)
SELECT location, founded_year, team_name FROM team WHERE (founded_year > 1989 AND ownership != 'Gabe Plotkin and Rick Schnall') OR (ownership = ' ' AND founded_year = 1967);

