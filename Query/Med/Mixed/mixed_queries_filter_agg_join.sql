-- Query 1: filter1_agg1_join1 (disease, drug)
SELECT drug.prescription_status, MIN(drug.recommended_usage) AS min_drug_recommended_usage FROM disease JOIN drug ON disease.disease_name = drug.disease_name WHERE drug.manufacturer != 'GSK' GROUP BY drug.prescription_status;

-- Query 2: filter2_agg1_join2 (disease, drug, institution)
SELECT institution.research_fields, MAX(disease.quality_of_life_impact) AS max_disease_quality_of_life_impact FROM disease JOIN drug ON disease.disease_name = drug.disease_name JOIN institution ON disease.disease_name = institution.research_diseases WHERE institution.leadership = 'Pierre-Yves Marcy' AND drug.activation_conditions != 'take with food' GROUP BY institution.research_fields;

-- Query 3: filter3_agg1_join1 (disease, drug)
SELECT drug.prescription_status, AVG(disease.quality_of_life_impact) AS avg_disease_quality_of_life_impact FROM disease JOIN drug ON disease.disease_name = drug.disease_name WHERE disease.treatment_challenges != 'only_one_drug_available' OR disease.drugs != 'diuretics' GROUP BY drug.prescription_status;

-- Query 4: filter4_agg1_join2 (disease, drug, institution)
SELECT institution.research_fields, MAX(disease.diagnosis_challenges) AS max_disease_diagnosis_challenges FROM disease JOIN drug ON disease.disease_name = drug.disease_name JOIN institution ON disease.disease_name = institution.research_diseases WHERE institution.institution_country = 'United Kingdom' AND institution.research_fields != 'gastroenterology' AND disease.etiology = 'pulmonary embolism' GROUP BY institution.research_fields;

-- Query 5: filter5_agg1_join1 (disease, drug)
SELECT drug.prescription_status, AVG(disease.quality_of_life_impact) AS avg_disease_quality_of_life_impact FROM disease JOIN drug ON disease.disease_name = drug.disease_name WHERE drug.brand_name = 'Nucala' OR disease.diagnostic_methods = 'genetic_testing' OR drug.activation_conditions = 'requires co-administration with other drugs' GROUP BY drug.prescription_status;

-- Query 6: filter6_agg1_join2 (disease, drug, institution)
SELECT institution.institution_type, AVG(disease.epidemiology) AS avg_disease_epidemiology FROM disease JOIN drug ON disease.disease_name = drug.disease_name JOIN institution ON disease.disease_name = institution.research_diseases WHERE (drug.brand_name != 'Daliresp' AND disease.common_symptoms != 'back pain') OR (institution.key_technologies = 'enzyme-linked immunosorbent assay (ELISA)' AND drug.pharmaceutical_form != 'ointment') GROUP BY institution.institution_type;

-- Query 7: filter1_agg1_join1 (disease, drug)
SELECT drug.prescription_status, AVG(disease.epidemiology) AS avg_disease_epidemiology FROM disease JOIN drug ON disease.disease_name = drug.disease_name WHERE drug.pharmaceutical_form = 'tablet' GROUP BY drug.prescription_status;

-- Query 8: filter2_agg1_join2 (disease, drug, institution)
SELECT institution.institution_type, AVG(drug.dosage_frequency) AS avg_drug_dosage_frequency FROM disease JOIN drug ON disease.disease_name = drug.disease_name JOIN institution ON disease.disease_name = institution.research_diseases WHERE drug.generic_name != 'Linaclotide' AND disease.common_symptoms != 'headache' GROUP BY institution.institution_type;

-- Query 9: filter3_agg1_join1 (disease, drug)
SELECT drug.prescription_status, COUNT(disease.common_symptoms) AS count_disease_common_symptoms FROM disease JOIN drug ON disease.disease_name = drug.disease_name WHERE disease.diagnostic_methods = 'genetic_testing' OR disease.pathogenesis = 'traumatic' GROUP BY drug.prescription_status;

-- Query 10: filter4_agg1_join2 (disease, drug, institution)
SELECT drug.prescription_status, SUM(institution.institution_country) AS sum_institution_institution_country FROM disease JOIN drug ON disease.disease_name = drug.disease_name JOIN institution ON disease.disease_name = institution.research_diseases WHERE disease.diagnosis_challenges = 'lack of early recognition' AND disease.disease_name = 'Acute Intermittent Porphyria' AND disease.epidemiology = '27% of constipated patients relate constipation to medications' GROUP BY drug.prescription_status;

-- Query 11: filter5_agg1_join1 (disease, drug)
SELECT drug.prescription_status, SUM(disease.treatment_challenges) AS sum_disease_treatment_challenges FROM disease JOIN drug ON disease.disease_name = drug.disease_name WHERE disease.treatments != 'radiotherapy' OR drug.brand_name != 'Lo Loestrin FE' OR drug.brand_name = 'Nucala' GROUP BY drug.prescription_status;

-- Query 12: filter6_agg1_join2 (disease, drug, institution)
SELECT institution.institution_type, MIN(disease.diagnosis_challenges) AS min_disease_diagnosis_challenges FROM disease JOIN drug ON disease.disease_name = drug.disease_name JOIN institution ON disease.disease_name = institution.research_diseases WHERE (drug.side_effects = 'diarrhea' AND drug.generic_name = 'Esomeprazole') OR (institution.research_diseases = 'Gut Microbiome Imbalance' AND disease.pathogenesis = 'infectious_fungal') GROUP BY institution.institution_type;

