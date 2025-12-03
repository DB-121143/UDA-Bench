-- Query 1: filter1_agg1 (institution)
SELECT research_fields, MIN(institution_country) AS min_institution_country FROM institution WHERE research_diseases != 'post-traumatic stress disorder' GROUP BY research_fields;

-- Query 2: filter2_agg1 (institution)
SELECT institution_type, SUM(institution_country) AS sum_institution_country FROM institution WHERE parent_organization = 'University of Wisconsin' AND funding_sources != 'other' GROUP BY institution_type;

-- Query 3: filter3_agg1 (drug)
SELECT prescription_status, COUNT(manufacturer) AS count_manufacturer FROM drug WHERE brand_name = 'Savella' OR recommended_usage >= 'oral' GROUP BY prescription_status;

-- Query 4: filter4_agg1 (drug)
SELECT prescription_status, MAX(dosage_frequency) AS max_dosage_frequency FROM drug WHERE mechanism_of_action != 'competitive inhibition of HMG-CoA reductase, reduces cholesterol synthesis, increases LDL receptor expression, increases LDL catabolism, inhibits VLDL synthesis' AND mechanism_of_action = 'relieve watery eyes, itchy eyes/nose/throat, runny nose, and sneezing' AND pharmaceutical_form != 'gel' GROUP BY prescription_status;

-- Query 5: filter5_agg1 (drug)
SELECT prescription_status, AVG(storage_conditions) AS avg_storage_conditions FROM drug WHERE single_dose = 'adults 25 mg' OR unsuitable_population != 'patients with diabetes' OR recommended_usage != 'intravenous' GROUP BY prescription_status;

-- Query 6: filter6_agg1 (drug)
SELECT prescription_status, MIN(storage_conditions) AS min_storage_conditions FROM drug WHERE (prescription_status < 'unclassified' AND side_effects != 'hair loss') OR (disease_name != 'Chronic Obstructive Pulmonary Disease' AND administration_route = 'intravenous') GROUP BY prescription_status;

