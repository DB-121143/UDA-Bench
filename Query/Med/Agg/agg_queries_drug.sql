-- Query 1: aggregation (drug)
SELECT prescription_status, MIN(dosage_frequency) AS min_dosage_frequency FROM drug GROUP BY prescription_status;

-- Query 2: aggregation (drug)
SELECT prescription_status, COUNT(generic_name) AS count_generic_name FROM drug GROUP BY prescription_status;

-- Query 3: aggregation (drug)
SELECT prescription_status, AVG(recommended_usage) AS avg_recommended_usage FROM drug GROUP BY prescription_status;

-- Query 4: aggregation (drug)
SELECT prescription_status, MAX(recommended_usage) AS max_recommended_usage FROM drug GROUP BY prescription_status;

-- Query 5: aggregation (drug)
SELECT prescription_status, COUNT(storage_conditions) AS count_storage_conditions FROM drug GROUP BY prescription_status;

-- Query 6: aggregation (drug)
SELECT prescription_status, MIN(recommended_usage) AS min_recommended_usage FROM drug GROUP BY prescription_status;

-- Query 7: aggregation (drug)
SELECT prescription_status, MAX(recommended_usage) AS max_recommended_usage FROM drug GROUP BY prescription_status;

-- Query 8: aggregation (drug)
SELECT prescription_status, SUM(recommended_usage) AS sum_recommended_usage FROM drug GROUP BY prescription_status;

