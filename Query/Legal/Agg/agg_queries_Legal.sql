-- Query 1: aggregation (Legal)
SELECT verdict, MIN(case_number) AS min_case_number FROM Legal GROUP BY verdict;

-- Query 2: aggregation (Legal)
SELECT nationality_for_applicant, COUNT(judge_name) AS count_judge_name FROM Legal GROUP BY nationality_for_applicant;

-- Query 3: aggregation (Legal)
SELECT nationality_for_applicant, AVG(legal_basis_num) AS avg_legal_basis_num FROM Legal GROUP BY nationality_for_applicant;

-- Query 4: aggregation (Legal)
SELECT nationality_for_applicant, MAX(legal_basis_num) AS max_legal_basis_num FROM Legal GROUP BY nationality_for_applicant;

-- Query 5: aggregation (Legal)
SELECT nationality_for_applicant, COUNT(defendant_current_status) AS count_defendant_current_status FROM Legal GROUP BY nationality_for_applicant;

-- Query 6: aggregation (Legal)
SELECT verdict, MIN(legal_basis_num) AS min_legal_basis_num FROM Legal GROUP BY verdict;

-- Query 7: aggregation (Legal)
SELECT nationality_for_applicant, MAX(legal_basis_num) AS max_legal_basis_num FROM Legal GROUP BY nationality_for_applicant;

-- Query 8: aggregation (Legal)
SELECT case_type, SUM(legal_basis_num) AS sum_legal_basis_num FROM Legal GROUP BY case_type;

-- Query 9: aggregation (Legal)
SELECT case_type, COUNT(evidence) AS count_evidence FROM Legal GROUP BY case_type;

-- Query 10: aggregation (Legal)
SELECT case_type, COUNT(case_number) AS count_case_number FROM Legal GROUP BY case_type;

