-- Query 1: binary_join (drug, disease)
SELECT disease.diagnostic_methods, drug.manufacturer, drug.brand_name, disease.disease_name FROM drug JOIN disease ON drug.disease_name = disease.disease_name;

-- Query 2: binary_join (drug, disease)
SELECT drug.mechanism_of_action, drug.storage_conditions, disease.diagnosis_challenges, disease.disease_name FROM drug JOIN disease ON drug.disease_name = disease.disease_name;

-- Query 3: binary_join (drug, disease)
SELECT drug.single_dose, disease.pathogenesis, drug.recommended_usage, disease.diagnosis_challenges FROM drug JOIN disease ON drug.disease_name = disease.disease_name;

-- Query 4: binary_join (drug, disease)
SELECT disease.treatments, disease.epidemiology, drug.prescription_status, drug.storage_conditions FROM drug JOIN disease ON drug.disease_name = disease.disease_name;

-- Query 5: binary_join (drug, disease)
SELECT disease.diagnostic_methods, disease.preventive_measures, drug.mechanism_of_action, drug.recommended_usage FROM drug JOIN disease ON drug.disease_name = disease.disease_name;

-- Query 6: binary_join (drug, disease)
SELECT drug.manufacturer, disease.complications, drug.pharmaceutical_form, disease.preventive_measures FROM drug JOIN disease ON drug.disease_name = disease.disease_name;

-- Query 7: binary_join (drug, disease)
SELECT drug.brand_name, drug.unsuitable_population, disease.preventive_measures, disease.diagnosis_challenges FROM drug JOIN disease ON drug.disease_name = disease.disease_name;

-- Query 8: binary_join (drug, disease)
SELECT drug.administration_route, disease.pathogenesis, disease.etiology, drug.mechanism_of_action FROM drug JOIN disease ON drug.disease_name = disease.disease_name;

-- Query 9: binary_join (drug, disease)
SELECT disease.epidemiology, disease.sequelae, drug.indication, drug.activation_conditions FROM drug JOIN disease ON drug.disease_name = disease.disease_name;

-- Query 10: binary_join (drug, disease)
SELECT disease.treatment_challenges, disease.diagnostic_methods, drug.administration_route, drug.generic_name FROM drug JOIN disease ON drug.disease_name = disease.disease_name;

-- Query 11: multi_table_join (drug, disease, institution)
SELECT disease.treatments, institution.establishment_year, institution.parent_organization, institution.number_of_staff, drug.unsuitable_population FROM drug JOIN disease ON drug.disease_name = disease.disease_name JOIN institution ON disease.disease_name = institution.research_diseases;

-- Query 12: multi_table_join (drug, disease, institution)
SELECT institution.research_diseases, disease.risk_factors, drug.mechanism_of_action, disease.disease_name, drug.unsuitable_population FROM drug JOIN disease ON drug.disease_name = disease.disease_name JOIN institution ON disease.disease_name = institution.research_diseases;

-- Query 13: multi_table_join (drug, disease, institution)
SELECT institution.key_technologies, disease.etiology, drug.indication, disease.prognosis, drug.unsuitable_population FROM drug JOIN disease ON drug.disease_name = disease.disease_name JOIN institution ON disease.disease_name = institution.research_diseases;

-- Query 14: multi_table_join (drug, disease, institution)
SELECT drug.unsuitable_population, disease.disease_type, drug.recommended_usage, institution.institution_city, disease.sequelae FROM drug JOIN disease ON drug.disease_name = disease.disease_name JOIN institution ON disease.disease_name = institution.research_diseases;

-- Query 15: multi_table_join (drug, disease, institution)
SELECT disease.diagnostic_methods, disease.disease_type, institution.number_of_staff, drug.disease_name, institution.international_collaboration FROM drug JOIN disease ON drug.disease_name = disease.disease_name JOIN institution ON disease.disease_name = institution.research_diseases;

-- Query 16: multi_table_join (drug, disease, institution)
SELECT institution.key_achievements, drug.storage_conditions, disease.prognosis, institution.institution_country, institution.number_of_staff FROM drug JOIN disease ON drug.disease_name = disease.disease_name JOIN institution ON disease.disease_name = institution.research_diseases;

-- Query 17: multi_table_join (drug, disease, institution)
SELECT institution.technology_application, drug.mechanism_of_action, disease.epidemiology, drug.brand_name, disease.quality_of_life_impact FROM drug JOIN disease ON drug.disease_name = disease.disease_name JOIN institution ON disease.disease_name = institution.research_diseases;

-- Query 18: multi_table_join (drug, disease, institution)
SELECT disease.common_symptoms, institution.institution_city, institution.technology_application, disease.diagnostic_methods, drug.brand_name FROM drug JOIN disease ON drug.disease_name = disease.disease_name JOIN institution ON disease.disease_name = institution.research_diseases;

-- Query 19: multi_table_join (drug, disease, institution)
SELECT disease.epidemiology, disease.diagnosis_challenges, institution.institution_name, institution.key_technologies, drug.brand_name FROM drug JOIN disease ON drug.disease_name = disease.disease_name JOIN institution ON disease.disease_name = institution.research_diseases;

-- Query 20: multi_table_join (drug, disease, institution)
SELECT disease.epidemiology, disease.treatments, drug.pharmaceutical_form, institution.number_of_staff, institution.institution_type FROM drug JOIN disease ON drug.disease_name = disease.disease_name JOIN institution ON disease.disease_name = institution.research_diseases;

