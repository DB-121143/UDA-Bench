-- Query 1: filter1_agg1 (legal)
SELECT verdict, MIN(case_number) AS min_case_number FROM legal WHERE plaintiff_current_status != 'Activist' GROUP BY verdict;

-- Query 2: filter2_agg1 (legal)
SELECT nationality_for_applicant, MAX(case_number) AS max_case_number FROM legal WHERE defendant_current_status != 'Company' AND legal_basis_num >= 1 GROUP BY nationality_for_applicant;

-- Query 3: filter3_agg1 (legal)
SELECT nationality_for_applicant, MAX(legal_basis_num) AS max_legal_basis_num FROM legal WHERE verdict != 'Dismissed' OR first_judge != 0 GROUP BY nationality_for_applicant;

-- Query 4: filter4_agg1 (legal)
SELECT verdict, MIN(legal_basis_num) AS min_legal_basis_num FROM legal WHERE legal_fees = '3265' AND defendant != 'Construction, Forestry, Mining and Energy Union' AND legal_fees != '2000' GROUP BY verdict;

-- Query 5: filter5_agg1 (legal)
SELECT nationality_for_applicant, MAX(case_number) AS max_case_number FROM legal WHERE first_judge != 0 OR evidence < 1 OR defendant = 'Secretary, Department of Employment and Workplace Relations' GROUP BY nationality_for_applicant;

-- Query 6: filter6_agg1 (legal)
SELECT case_type, SUM(legal_basis_num) AS sum_legal_basis_num FROM legal WHERE (judgment_year = 2009 AND plaintiff != 'Telstra Corporation Limited') OR (verdict != 'Others' AND counsel_for_applicant = 'Dr J G Azzi') GROUP BY case_type;

