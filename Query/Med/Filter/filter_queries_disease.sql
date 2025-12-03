-- Query 1: 1 (disease)
SELECT epidemiology, risk_factors, etiology FROM disease WHERE risk_factors != 'infection';

-- Query 2: 1 (disease)
SELECT sequelae, diagnostic_methods, affected_organs FROM disease WHERE affected_organs = 'ligaments';

-- Query 3: 1 (disease)
SELECT treatment_challenges, disease_type, risk_factors FROM disease WHERE disease_type != 'others';

-- Query 4: 1 (disease)
SELECT affected_organs, pathogenesis, treatments FROM disease WHERE treatments = 'stem cell transplantation';

-- Query 5: 2 (disease)
SELECT diagnosis_challenges, preventive_measures, disease_type FROM disease WHERE preventive_measures != 'exercise' AND common_symptoms != 'nausea';

-- Query 6: 2 (disease)
SELECT diagnostic_methods, prognosis, affected_organs FROM disease WHERE prognosis = 'high_mortality' AND drugs = 'phenytoin';

-- Query 7: 2 (disease)
SELECT risk_factors, prognosis, treatments FROM disease WHERE risk_factors != 'chronic pain' AND prognosis != 'asymptomatic';

-- Query 8: 2 (disease)
SELECT treatments, affected_organs, etiology FROM disease WHERE affected_organs != 'cervix' AND epidemiology = 'occurs most often among Caucasian women';

-- Query 9: 3 (disease)
SELECT diagnostic_methods, pathogenesis, drugs FROM disease WHERE diagnostic_methods != 'screening' OR drugs != 'acyclovir';

-- Query 10: 3 (disease)
SELECT diagnosis_challenges, affected_organs, preventive_measures FROM disease WHERE diagnosis_challenges != 'difficult to find early' OR treatment_challenges != 'no clear silver bullet for therapy';

-- Query 11: 3 (disease)
SELECT diagnostic_methods, common_symptoms, quality_of_life_impact FROM disease WHERE quality_of_life_impact = 'mobility_impairment' OR etiology != 'pneumonia';

-- Query 12: 3 (disease)
SELECT quality_of_life_impact, etiology, pathogenesis FROM disease WHERE pathogenesis = 'congenital' OR sequelae != 'chronic pain';

-- Query 13: 4 (disease)
SELECT quality_of_life_impact, risk_factors, preventive_measures FROM disease WHERE risk_factors != 'sedentary_lifestyle' AND affected_organs != 'bone' AND epidemiology != 'very common sexually transmitted infection' AND sequelae != 'blindness';

-- Query 14: 4 (disease)
SELECT treatment_challenges, common_symptoms, disease_type FROM disease WHERE disease_type = 'iatrogenic' AND common_symptoms != 'tenderness' AND complications != 'chlamydial conjunctivitis' AND treatments != 'immunotherapy';

-- Query 15: 4 (disease)
SELECT affected_organs, complications, diagnostic_methods FROM disease WHERE affected_organs = 'abdomen' AND complications != 'panic disorders' AND quality_of_life_impact != 'social_isolation' AND diagnostic_methods != 'clinical_evaluation';

-- Query 16: 4 (disease)
SELECT epidemiology, sequelae, etiology FROM disease WHERE epidemiology != '3,500 to 4,000 children worldwide; up to 80% mortality' AND disease_type != 'autoimmune' AND epidemiology = 'reported in almost every state' AND disease_name != 'Chlamydia';

-- Query 17: 5 (disease)
SELECT epidemiology, affected_organs, sequelae FROM disease WHERE affected_organs != 'toes' OR etiology = 'genetic predisposition' OR treatment_challenges != 'limited options' OR pathogenesis != 'infectious_fungal';

-- Query 18: 5 (disease)
SELECT treatment_challenges, disease_type, complications FROM disease WHERE complications = 'cervicitis' OR diagnostic_methods != 'laboratory_test' OR disease_name = 'Opioid Addiction' OR complications = 'heart related complications';

-- Query 19: 5 (disease)
SELECT disease_type, drugs, treatment_challenges FROM disease WHERE disease_type != 'allergic' OR diagnostic_methods = 'endoscopy' OR drugs != 'antidepressants' OR preventive_measures = 'vaccination';

-- Query 20: 5 (disease)
SELECT diagnosis_challenges, complications, diagnostic_methods FROM disease WHERE complications != 'large_blood_vessel_damage' OR disease_name != 'Pleural Effusion' OR treatment_challenges = 'financial_burden' OR quality_of_life_impact = 'dietary_restriction';

-- Query 21: 6 (disease)
SELECT treatments, common_symptoms, quality_of_life_impact FROM disease WHERE (common_symptoms = 'constipation' AND prognosis = 'variable') OR (epidemiology != '741 DR-TB cases in Ethiopia in 2018' AND risk_factors != 'immunosuppression');

-- Query 22: 6 (disease)
SELECT disease_name, preventive_measures, common_symptoms FROM disease WHERE (common_symptoms != 'decreased appetite' AND complications != 'cardiovascular_disorders') OR (disease_type = 'infectious' AND disease_type = 'genetic');

-- Query 23: 6 (disease)
SELECT affected_organs, quality_of_life_impact, sequelae FROM disease WHERE (affected_organs = 'nervous system' AND risk_factors != 'hyperlipidemia') OR (etiology = 'insulin resistance' AND diagnostic_methods != 'screening');

-- Query 24: 6 (disease)
SELECT diagnostic_methods, disease_type, diagnosis_challenges FROM disease WHERE (diagnosis_challenges = 'difficult to define onset after opioid initiation' AND prognosis = 'stable') OR (risk_factors != 'genetic_predisposition' AND diagnostic_methods = 'genetic_testing');

