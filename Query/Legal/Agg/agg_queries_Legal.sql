-- Query 1: aggregation (legal)
SELECT verdict, MIN(case_number) AS min_case_number FROM legal GROUP BY verdict;

-- Query 2: aggregation (legal)
SELECT nationality_for_applicant, COUNT(judge_name) AS count_judge_name FROM legal GROUP BY nationality_for_applicant;

-- Query 3: aggregation (legal)
SELECT nationality_for_applicant, AVG(legal_basis_num) AS avg_legal_basis_num FROM legal GROUP BY nationality_for_applicant;

-- Query 4: aggregation (legal)
SELECT nationality_for_applicant, MAX(legal_basis_num) AS max_legal_basis_num FROM legal GROUP BY nationality_for_applicant;

-- Query 5: aggregation (legal)
SELECT nationality_for_applicant, COUNT(defendant_current_status) AS count_defendant_current_status FROM legal GROUP BY nationality_for_applicant;

-- Query 6: aggregation (legal)
SELECT verdict, MIN(legal_basis_num) AS min_legal_basis_num FROM legal GROUP BY verdict;

-- Query 7: aggregation (legal)
SELECT nationality_for_applicant, MAX(legal_basis_num) AS max_legal_basis_num FROM legal GROUP BY nationality_for_applicant;

-- Query 8: aggregation (legal)
SELECT case_type, SUM(legal_basis_num) AS sum_legal_basis_num FROM legal GROUP BY case_type;

-- Query 9: aggregation (legal)
SELECT case_type, COUNT(evidence) AS count_evidence FROM legal GROUP BY case_type;

-- Query 10: aggregation (legal)
SELECT case_type, COUNT(case_number) AS count_case_number FROM legal GROUP BY case_type;

