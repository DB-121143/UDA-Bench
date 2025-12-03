-- Query 1: aggregation (institution)
SELECT research_fields, MIN(institution_country) AS min_institution_country FROM institution GROUP BY research_fields;

-- Query 2: aggregation (institution)
SELECT institution_type, COUNT(institution_name) AS count_institution_name FROM institution GROUP BY institution_type;

