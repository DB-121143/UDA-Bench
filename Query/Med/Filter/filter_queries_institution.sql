-- Query 1: 1 (institution)
SELECT key_technologies, institution_country, international_collaboration FROM institution WHERE technology_application != 'FDA drug approval';

-- Query 2: 1 (institution)
SELECT leadership, parent_organization, establishment_year FROM institution WHERE leadership = 'Tanja Gruber';

-- Query 3: 2 (institution)
SELECT institution_city, international_collaboration, institution_name FROM institution WHERE research_fields != 'molecular_biology' AND funding_sources = 'university grant';

-- Query 4: 2 (institution)
SELECT leadership, research_fields, international_collaboration FROM institution WHERE technology_application = 'early diagnosis of cardiac TTR amyloidosis using DPD scintigraphy' AND key_technologies = 'navigation for insertion of screws';

-- Query 5: 3 (institution)
SELECT number_of_staff, parent_organization, key_technologies FROM institution WHERE international_collaboration = 'NCI cancer centers in the United States' OR institution_country != 'China';

-- Query 6: 3 (institution)
SELECT institution_city, establishment_year, technology_application FROM institution WHERE research_fields != 'molecular_biology' OR international_collaboration != 'collaboration with research centres throughout Europe and the UK';

-- Query 7: 4 (institution)
SELECT technology_application, institution_country, institution_city FROM institution WHERE research_diseases = 'portal hypertensive gastropathy' AND key_technologies != 'bacteriological culture' AND research_diseases != 'chronic pain' AND institution_city != 'Ferndale';

-- Query 8: 4 (institution)
SELECT institution_name, research_fields, research_diseases FROM institution WHERE key_achievements = 'post-graduate education in gastroenterology' AND parent_organization != 'University of Texas' AND leadership != 'K Persson Waller' AND institution_country > 'Vietnam';

-- Query 9: 5 (institution)
SELECT establishment_year, international_collaboration, leadership FROM institution WHERE leadership != 'John Niederhuber' OR research_fields != 'surgery' OR institution_name != 'Emory University' OR key_technologies = 'antibacterial photodynamic treatment (aPDT)';

-- Query 10: 5 (institution)
SELECT international_collaboration, establishment_year, key_achievements FROM institution WHERE leadership = 'K Persson Waller' OR parent_organization != 'UNSW Sydney' OR institution_name = 'Flushing Hospital Medical Center' OR technology_application != 'individual counseling';

-- Query 11: 6 (institution)
SELECT number_of_staff, parent_organization, technology_application FROM institution WHERE (parent_organization = 'Shanghai Jiao Tong University School of Medicine' AND institution_city = 'London') OR (research_diseases != 'lower limb complications of diabetes' AND technology_application != 'intensive case management');

-- Query 12: 6 (institution)
SELECT parent_organization, key_achievements, international_collaboration FROM institution WHERE (technology_application = 'inpatient psychiatric treatment' AND key_technologies = 'blood sample collection') OR (funding_sources != 'industry grant' AND key_achievements = 'Largest neuroscience study of mindfulness for addiction');

