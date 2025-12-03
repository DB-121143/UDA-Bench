-- Query 1: aggregation (Legal)
SELECT verdict, legal_fees, fine_amount, MIN(case_number) AS min_case_number FROM Legal GROUP BY verdict;

-- Query 2: aggregation (Legal)
SELECT nationality_for_applicant, hearing_year, judge_name, COUNT(judge_name) AS count_judge_name FROM Legal GROUP BY nationality_for_applicant;

-- Query 3: aggregation (Legal)
SELECT nationality_for_applicant, case_number, plaintiff, AVG(legal_basis_num) AS avg_legal_basis_num FROM Legal GROUP BY nationality_for_applicant;

-- Query 4: aggregation (Legal)
SELECT nationality_for_applicant, plaintiff_current_status, plaintiff, MAX(legal_basis_num) AS max_legal_basis_num FROM Legal GROUP BY nationality_for_applicant;

-- Query 5: aggregation (Legal)
SELECT nationality_for_applicant, hearing_year, evidence, COUNT(defendant_current_status) AS count_defendant_current_status FROM Legal GROUP BY nationality_for_applicant;

-- Query 6: aggregation (Legal)
SELECT verdict, fine_amount, judge_name, MIN(legal_basis_num) AS min_legal_basis_num FROM Legal GROUP BY verdict;

-- Query 7: aggregation (Legal)
SELECT nationality_for_applicant, case_number, judge_name, MAX(legal_basis_num) AS max_legal_basis_num FROM Legal GROUP BY nationality_for_applicant;

-- Query 8: aggregation (Legal)
SELECT case_type, charges, plaintiff, SUM(legal_basis_num) AS sum_legal_basis_num FROM Legal GROUP BY case_type;

-- Query 9: aggregation (Legal)
SELECT case_type, verdict, judge_name, COUNT(evidence) AS count_evidence FROM Legal GROUP BY case_type;

-- Query 10: aggregation (Legal)
SELECT case_type, judge_name, counsel_for_applicant, COUNT(case_number) AS count_case_number FROM Legal GROUP BY case_type;

