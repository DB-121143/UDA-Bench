-- Query 1: filter1_agg1 (Legal)
SELECT verdict, MIN(case_number) AS min_case_number FROM Legal WHERE plaintiff_current_status != 'Activist' GROUP BY verdict;

-- Query 2: filter2_agg1 (Legal)
SELECT nationality_for_applicant, MAX(case_number) AS max_case_number FROM Legal WHERE defendant_current_status != 'Company' AND legal_basis_num >= 1 GROUP BY nationality_for_applicant;

-- Query 3: filter3_agg1 (Legal)
SELECT nationality_for_applicant, MAX(legal_basis_num) AS max_legal_basis_num FROM Legal WHERE verdict != 'Dismissed' OR first_judge != 0 GROUP BY nationality_for_applicant;

-- Query 4: filter4_agg1 (Legal)
SELECT verdict, MIN(legal_basis_num) AS min_legal_basis_num FROM Legal WHERE legal_fees = '3265' AND defendant != 'Construction, Forestry, Mining and Energy Union' AND legal_fees != '2000' GROUP BY verdict;

-- Query 5: filter5_agg1 (Legal)
SELECT nationality_for_applicant, MAX(case_number) AS max_case_number FROM Legal WHERE first_judge != 0 OR evidence < 1 OR defendant = 'Secretary, Department of Employment and Workplace Relations' GROUP BY nationality_for_applicant;

-- Query 6: filter6_agg1 (Legal)
SELECT case_type, SUM(legal_basis_num) AS sum_legal_basis_num FROM Legal WHERE (judgment_year = 2009 AND plaintiff != 'Telstra Corporation Limited') OR (verdict != 'Others' AND counsel_for_applicant = 'Dr J G Azzi') GROUP BY case_type;

