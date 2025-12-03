-- Query 1: 1 (drug)
SELECT manufacturer, activation_conditions, indication FROM drug WHERE activation_conditions = 'requires stimulation';

-- Query 2: 1 (drug)
SELECT pharmaceutical_form, active_ingredients, administration_route FROM drug WHERE administration_route = 'injection';

-- Query 3: 1 (drug)
SELECT manufacturer, brand_name, activation_conditions FROM drug WHERE brand_name != 'Actipraz 40 Mg Tablet';

-- Query 4: 1 (drug)
SELECT indication, disease_name, recommended_usage FROM drug WHERE recommended_usage = 'take as prescribed';

-- Query 5: 2 (drug)
SELECT prescription_status, side_effects, brand_name FROM drug WHERE prescription_status < 'unclassified' AND pharmaceutical_form != 'capsule';

-- Query 6: 2 (drug)
SELECT recommended_usage, administration_route, dosage_frequency FROM drug WHERE dosage_frequency > 'every 4 to 6 hours as needed' AND active_ingredients != 'azithromycin';

-- Query 7: 2 (drug)
SELECT activation_conditions, single_dose, recommended_usage FROM drug WHERE activation_conditions != 'take with food' AND dosage_frequency <= 'every 8 hours';

-- Query 8: 2 (drug)
SELECT administration_route, indication, recommended_usage FROM drug WHERE administration_route != 'injection' AND storage_conditions < 'protect from light';

-- Query 9: 3 (drug)
SELECT active_ingredients, brand_name, single_dose FROM drug WHERE active_ingredients != 'Antihistamines' OR single_dose != 'children 2-5 years 36 mcg';

-- Query 10: 3 (drug)
SELECT unsuitable_population, administration_route, storage_conditions FROM drug WHERE unsuitable_population != 'patients with cyst on the ovary' OR storage_conditions > 'avoid freezing';

-- Query 11: 3 (drug)
SELECT brand_name, pharmaceutical_form, active_ingredients FROM drug WHERE active_ingredients = 'mepolizumab' OR activation_conditions != 'before meals';

-- Query 12: 3 (drug)
SELECT indication, prescription_status, disease_name FROM drug WHERE disease_name = 'Parkinson''s disease' OR mechanism_of_action != 'improving metabolic profile';

-- Query 13: 4 (drug)
SELECT prescription_status, activation_conditions, storage_conditions FROM drug WHERE activation_conditions = 'before meals' AND storage_conditions < 'protect from moisture' AND side_effects != 'headache' AND mechanism_of_action != 'induces loss of consciousness';

-- Query 14: 4 (drug)
SELECT disease_name, pharmaceutical_form, brand_name FROM drug WHERE brand_name = 'Omlek-D' AND pharmaceutical_form = 'gel' AND disease_name != 'Herpes simplex' AND indication != 'infections';

-- Query 15: 4 (drug)
SELECT mechanism_of_action, active_ingredients, administration_route FROM drug WHERE administration_route = 'injection' AND manufacturer != 'Astra Zeneca' AND mechanism_of_action != 'enhancing glucose metabolism' AND administration_route != 'oral';

-- Query 16: 4 (drug)
SELECT single_dose, mechanism_of_action, side_effects FROM drug WHERE side_effects = 'fever' AND storage_conditions = 'store below 25C or 77F' AND mechanism_of_action != 'broken down into 6-thioguanine nucleotides to exert immunosuppressive effects' AND active_ingredients = 'Omeprazole';

-- Query 17: 5 (drug)
SELECT administration_route, side_effects, mechanism_of_action FROM drug WHERE administration_route != 'injection' OR indication = 'heartburn' OR storage_conditions > 'store in a tight, light-resistant container' OR recommended_usage = 'other';

-- Query 18: 5 (drug)
SELECT brand_name, manufacturer, disease_name FROM drug WHERE manufacturer = 'Astellas' OR active_ingredients != 'Pravastatin sodium salt' OR generic_name = 'Pramipexole dihydrochloride' OR manufacturer = 'Preferred Pharmaceuticals, Inc.';

-- Query 19: 5 (drug)
SELECT brand_name, single_dose, active_ingredients FROM drug WHERE brand_name != 'Duaklir Genuair' OR active_ingredients = 'Milnacipran' OR single_dose != 'adults 10-40 mg once a day' OR prescription_status = 'unclassified';

-- Query 20: 5 (drug)
SELECT administration_route, manufacturer, active_ingredients FROM drug WHERE manufacturer != 'Cipla' OR generic_name != 'Ropinirole' OR storage_conditions >= 'store at room temperature' OR brand_name = 'Omlek-D';

-- Query 21: 6 (drug)
SELECT recommended_usage, pharmaceutical_form, disease_name FROM drug WHERE (pharmaceutical_form = 'spray' AND dosage_frequency = 'twice daily') OR (side_effects != 'weight loss' AND activation_conditions != 'before meals');

-- Query 22: 6 (drug)
SELECT recommended_usage, pharmaceutical_form, generic_name FROM drug WHERE (pharmaceutical_form != 'granule' AND manufacturer != 'Avanir Pharmaceuticals') OR (brand_name = 'PEG-Intron' AND brand_name = 'Nexium');

-- Query 23: 6 (drug)
SELECT unsuitable_population, mechanism_of_action, administration_route FROM drug WHERE (administration_route = 'nasal' AND activation_conditions != 'requires stimulation') OR (indication = 'severe asthma' AND active_ingredients != 'Psilocybin');

-- Query 24: 6 (drug)
SELECT disease_name, brand_name, unsuitable_population FROM drug WHERE (unsuitable_population = 'patients with stomach/intestinal problems' AND dosage_frequency = 'every 8 hours') OR (activation_conditions = 'requires stimulation' AND mechanism_of_action != 'produces relaxation to relieve pain');

