-- Query 1: filter1_join1 (disease, drug)
SELECT disease.epidemiology, drug.brand_name, disease.risk_factors, drug.recommended_usage FROM disease JOIN drug ON disease.disease_name = drug.disease_name WHERE drug.manufacturer != 'GSK';

-- Query 2: filter2_join2 (disease, drug, institution)
SELECT institution.key_achievements, drug.manufacturer, institution.funding_sources, disease.disease_type FROM disease JOIN drug ON disease.disease_name = drug.disease_name JOIN institution ON disease.disease_name = institution.research_diseases WHERE disease.pathogenesis = 'congenital' AND institution.institution_country >= 'Egypt';

-- Query 3: filter3_join1 (disease, drug)
SELECT disease.drugs, drug.disease_name, disease.treatments, drug.unsuitable_population FROM disease JOIN drug ON disease.disease_name = drug.disease_name WHERE drug.brand_name = 'REQUIP XL' OR drug.activation_conditions != 'before meals';

-- Query 4: filter4_join2 (disease, drug, institution)
SELECT drug.side_effects, disease.treatments, institution.key_technologies, drug.mechanism_of_action FROM disease JOIN drug ON disease.disease_name = drug.disease_name JOIN institution ON disease.disease_name = institution.research_diseases WHERE disease.treatment_challenges != 'only_one_drug_available' AND disease.drugs != 'diuretics' AND drug.administration_route = 'injection';

-- Query 5: filter5_join1 (disease, drug)
SELECT disease.disease_type, disease.diagnosis_challenges, drug.prescription_status, drug.unsuitable_population FROM disease JOIN drug ON disease.disease_name = drug.disease_name WHERE drug.activation_conditions != 'no special condition' OR drug.dosage_frequency >= 'once or twice a day' OR disease.pathogenesis = 'endocrine_disorder';

-- Query 6: filter6_join2 (disease, drug, institution)
SELECT institution.institution_country, drug.manufacturer, drug.indication, disease.risk_factors FROM disease JOIN drug ON disease.disease_name = drug.disease_name JOIN institution ON disease.disease_name = institution.research_diseases WHERE (drug.single_dose != 'adults 10 mg' AND drug.active_ingredients = 'Roflumilast') OR (drug.storage_conditions != 'do not freeze liquid forms' AND institution.research_fields != 'cardiology');

-- Query 7: filter1_join1 (disease, drug)
SELECT disease.drugs, drug.disease_name, drug.indication, disease.prognosis FROM disease JOIN drug ON disease.disease_name = drug.disease_name WHERE drug.brand_name = 'Nucala';

-- Query 8: filter2_join2 (disease, drug, institution)
SELECT disease.common_symptoms, drug.pharmaceutical_form, institution.funding_sources, drug.dosage_frequency FROM disease JOIN drug ON disease.disease_name = drug.disease_name JOIN institution ON disease.disease_name = institution.research_diseases WHERE disease.prognosis != 'partial_recovery' AND drug.active_ingredients != 'bendamustine';

-- Query 9: filter3_join1 (disease, drug)
SELECT drug.active_ingredients, disease.disease_name, disease.sequelae, drug.side_effects FROM disease JOIN drug ON disease.disease_name = drug.disease_name WHERE disease.disease_name = 'Metastatic Colorectal Cancer' OR drug.administration_route != 'injection';

-- Query 10: filter4_join2 (disease, drug, institution)
SELECT disease.sequelae, drug.brand_name, drug.pharmaceutical_form, institution.parent_organization FROM disease JOIN drug ON disease.disease_name = drug.disease_name JOIN institution ON disease.disease_name = institution.research_diseases WHERE drug.pharmaceutical_form = 'tablet' AND drug.indication != 'Runny nose' AND disease.preventive_measures = 'meditation';

-- Query 11: filter5_join1 (disease, drug)
SELECT drug.recommended_usage, disease.preventive_measures, drug.unsuitable_population, disease.disease_type FROM disease JOIN drug ON disease.disease_name = drug.disease_name WHERE drug.dosage_frequency < '2 or 3 times daily' OR disease.sequelae != 'disability' OR disease.epidemiology != 'especially amongst young men and women';

-- Query 12: filter6_join2 (disease, drug, institution)
SELECT drug.indication, institution.international_collaboration, disease.prognosis, drug.recommended_usage FROM disease JOIN drug ON disease.disease_name = drug.disease_name JOIN institution ON disease.disease_name = institution.research_diseases WHERE (drug.indication != 'constipation' AND disease.diagnosis_challenges != 'lack of accurate diagnostic test') OR (drug.single_dose = 'adult patients with EGPA 3x100 mg' AND disease.sequelae != 'gangrene');

