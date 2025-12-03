-- Query 1: agg1_join1 (disease, drug)
SELECT drug.prescription_status, MIN(drug.recommended_usage) AS min_drug_recommended_usage FROM disease JOIN drug ON disease.disease_name = drug.disease_name GROUP BY drug.prescription_status;

-- Query 2: agg1_join2 (disease, drug, institution)
SELECT institution.institution_type, MIN(drug.recommended_usage) AS min_drug_recommended_usage FROM disease JOIN drug ON disease.disease_name = drug.disease_name JOIN institution ON disease.disease_name = institution.research_diseases GROUP BY institution.institution_type;

