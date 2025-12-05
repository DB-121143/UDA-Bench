-- Query 1: 1 (legal)
SELECT nationality_for_applicant, fine_amount, hearing_year FROM legal WHERE fine_amount = 20000.00;

-- Query 2: 1 (legal)
SELECT counsel_for_respondent, judgment_year, verdict FROM legal WHERE verdict = 'Not Guilty';

-- Query 3: 1 (legal)
SELECT evidence, plaintiff, fine_amount FROM legal WHERE plaintiff != 'Deputy Commissioner of Taxation';

-- Query 4: 1 (legal)
SELECT verdict, defendant, legal_basis_num FROM legal WHERE legal_basis_num != 4;

-- Query 5: 1 (legal)
SELECT evidence, hearing_year, verdict FROM legal WHERE hearing_year = 2007;

-- Query 6: 1 (legal)
SELECT defendant_current_status, defendant, plaintiff_current_status FROM legal WHERE defendant = 'Darrell Lea Chocolate Shops Pty Ltd';

-- Query 7: 1 (legal)
SELECT case_number, plaintiff_current_status, plaintiff FROM legal WHERE case_number <= 5;

-- Query 8: 1 (legal)
SELECT verdict, counsel_for_respondent, plaintiff FROM legal WHERE counsel_for_respondent = 'Mr BJ Sullivan SC';

-- Query 9: 1 (legal)
SELECT hearing_year, legal_basis_num, defendant_current_status FROM legal WHERE defendant_current_status = 'Workplace Inspecto';

-- Query 10: 1 (legal)
SELECT legal_fees, judge_name, charges FROM legal WHERE judge_name != 'Black';

-- Query 11: 2 (legal)
SELECT plaintiff_current_status, legal_fees, plaintiff FROM legal WHERE legal_fees != '7500' AND charges != 'Abuse of process';

-- Query 12: 2 (legal)
SELECT defendant_current_status, verdict, counsel_for_applicant FROM legal WHERE counsel_for_applicant = 'M D Wyles' AND case_number >= 17;

-- Query 13: 2 (legal)
SELECT fine_amount, counsel_for_applicant, legal_basis_num FROM legal WHERE fine_amount != 210000.00 AND nationality_for_applicant != 'Bangladesh';

-- Query 14: 2 (legal)
SELECT verdict, hearing_year, defendant_current_status FROM legal WHERE verdict != 'Not Guilty' AND defendant_current_status != 'Company';

-- Query 15: 2 (legal)
SELECT case_number, first_judge, hearing_year FROM legal WHERE hearing_year = 2007 AND case_number >= 17;

-- Query 16: 2 (legal)
SELECT hearing_year, nationality_for_applicant, counsel_for_respondent FROM legal WHERE hearing_year != 2008 AND counsel_for_respondent = 'R Knowles';

-- Query 17: 2 (legal)
SELECT judgment_year, case_type, counsel_for_applicant FROM legal WHERE judgment_year != 2006 AND case_number < 17;

-- Query 18: 2 (legal)
SELECT nationality_for_applicant, plaintiff_current_status, verdict FROM legal WHERE verdict = 'Guilty' AND case_number >= 5;

-- Query 19: 2 (legal)
SELECT legal_fees, first_judge, defendant_current_status FROM legal WHERE legal_fees != '1300' AND legal_basis_num >= 4;

-- Query 20: 2 (legal)
SELECT nationality_for_applicant, counsel_for_applicant, hearing_year FROM legal WHERE hearing_year = 2008 AND plaintiff_current_status != 'Fisherman';

-- Query 21: 3 (legal)
SELECT judgment_year, defendant, case_number FROM legal WHERE judgment_year != 2006 OR case_number > 17;

-- Query 22: 3 (legal)
SELECT plaintiff_current_status, verdict, legal_fees FROM legal WHERE plaintiff_current_status != 'An Active Member Of The Bharatiya Janta Party In Kerala State' OR defendant_current_status != 'The trustee of Mr Boensch__ bankrupt estate';

-- Query 23: 3 (legal)
SELECT first_judge, judgment_year, evidence FROM legal WHERE evidence != 1 OR charges != 'Application forudicial review of a decision to deny a Remaining Relative visa';

-- Query 24: 3 (legal)
SELECT first_judge, hearing_year, defendant FROM legal WHERE defendant = 'Siminton' OR counsel_for_respondent != 'T Wong';

-- Query 25: 3 (legal)
SELECT evidence, fine_amount, nationality_for_applicant FROM legal WHERE fine_amount != 20000.00 OR defendant_current_status = 'Distribution of counterfeit goods saler';

-- Query 26: 3 (legal)
SELECT counsel_for_respondent, case_type, legal_basis_num FROM legal WHERE case_type != 'Commercial Case' OR judge_name != 'Gyles';

-- Query 27: 3 (legal)
SELECT verdict, fine_amount, legal_fees FROM legal WHERE verdict != 'Guilty' OR nationality_for_applicant = 'Philippines';

-- Query 28: 3 (legal)
SELECT counsel_for_respondent, case_type, defendant FROM legal WHERE defendant = 'IPM Operation and Maintenance Loy Yang Pty Ltd' OR judge_name != 'Weinbergj';

-- Query 29: 3 (legal)
SELECT verdict, nationality_for_applicant, counsel_for_respondent FROM legal WHERE verdict != 'Approved' OR evidence != 1;

-- Query 30: 3 (legal)
SELECT charges, counsel_for_respondent, nationality_for_applicant FROM legal WHERE counsel_for_respondent = 'Mr R Richter QC' OR hearing_year = 2007;

-- Query 31: 4 (legal)
SELECT first_judge, fine_amount, legal_fees FROM legal WHERE fine_amount = 210000.00 AND defendant_current_status != 'A workplace inspector' AND nationality_for_applicant != 'China' AND counsel_for_respondent != 'S Ricketson';

-- Query 32: 4 (legal)
SELECT plaintiff, first_judge, defendant_current_status FROM legal WHERE first_judge > 1 AND first_judge >= 0 AND charges != 'Administrative Case' AND case_type != 'Others';

-- Query 33: 4 (legal)
SELECT evidence, verdict, judgment_year FROM legal WHERE verdict = 'Not Guilty' AND case_type != 'Administrative Case' AND evidence >= 1 AND counsel_for_respondent != 'T Reilly';

-- Query 34: 4 (legal)
SELECT nationality_for_applicant, counsel_for_respondent, hearing_year FROM legal WHERE nationality_for_applicant != 'Vietnam' AND first_judge < 1 AND hearing_year = 2008 AND first_judge >= 0;

-- Query 35: 4 (legal)
SELECT defendant, hearing_year, nationality_for_applicant FROM legal WHERE hearing_year != 2006 AND defendant_current_status = 'Producer and distributor of the documentary' AND verdict = 'Approved' AND nationality_for_applicant = 'India';

-- Query 36: 4 (legal)
SELECT charges, nationality_for_applicant, case_number FROM legal WHERE charges != 'Sequestration orders against the estates of the applicants' AND nationality_for_applicant != 'Romania' AND hearing_year = 2009 AND hearing_year = 2008;

-- Query 37: 4 (legal)
SELECT first_judge, case_type, fine_amount FROM legal WHERE first_judge >= 1 AND first_judge <= 0 AND case_number < 10 AND defendant_current_status = 'Professor of Surgery in the University of Western Australia (UWA)';

-- Query 38: 4 (legal)
SELECT nationality_for_applicant, judge_name, defendant FROM legal WHERE judge_name != 'Dowsett' AND case_number = 17 AND counsel_for_applicant != 'Mr R Lilley SC' AND legal_basis_num >= 2;

-- Query 39: 4 (legal)
SELECT case_type, verdict, case_number FROM legal WHERE case_number != 17 AND plaintiff != 'Shen' AND legal_fees = '1300' AND verdict = 'Others';

-- Query 40: 4 (legal)
SELECT fine_amount, plaintiff_current_status, legal_basis_num FROM legal WHERE legal_basis_num >= 2 AND plaintiff_current_status != 'Bussiness Man' AND plaintiff_current_status != 'Employee' AND evidence <= 1;

-- Query 41: 5 (legal)
SELECT nationality_for_applicant, verdict, counsel_for_respondent FROM legal WHERE verdict != 'Guilty' OR hearing_year != 2007 OR plaintiff != 'Australian Prudential Regulation Authority' OR defendant_current_status = 'Professor of Surgery in the University of Western Australia (UWA)';

-- Query 42: 5 (legal)
SELECT defendant_current_status, plaintiff, case_type FROM legal WHERE case_type = 'Others' OR judgment_year = 2007 OR legal_fees != '2960' OR case_type != 'Others';

-- Query 43: 5 (legal)
SELECT first_judge, case_number, plaintiff FROM legal WHERE first_judge = 0 OR case_number != 10 OR verdict != 'Not Guilty' OR charges != 'Failure to consider the applicant''s refugee claims and failure to comply with s 424A(1)(a) of the Migration Act 1958';

-- Query 44: 5 (legal)
SELECT plaintiff_current_status, case_type, judgment_year FROM legal WHERE case_type != 'Others' OR judge_name != 'Greenwood' OR defendant_current_status = 'Company' OR evidence > 1;

-- Query 45: 5 (legal)
SELECT verdict, first_judge, case_number FROM legal WHERE verdict = 'Others' OR plaintiff_current_status = 'Deputy Commissioner Of Taxation' OR plaintiff_current_status != 'Employee' OR counsel_for_respondent != 'Mr J D Smith';

-- Query 46: 5 (legal)
SELECT counsel_for_respondent, first_judge, charges FROM legal WHERE counsel_for_respondent != 'Mr S Lloyd' OR fine_amount != 20000.00 OR plaintiff != 'Australian Communications and Media Authority' OR first_judge >= 1;

-- Query 47: 5 (legal)
SELECT fine_amount, judge_name, charges FROM legal WHERE fine_amount = 944120.00 OR case_number < 5 OR judge_name != 'Kiefel' OR fine_amount != 5000.00;

-- Query 48: 5 (legal)
SELECT fine_amount, evidence, hearing_year FROM legal WHERE hearing_year != 2007 OR plaintiff != 'Roadshow Films Pty Ltd' OR case_number > 10 OR case_number > 17;

-- Query 49: 5 (legal)
SELECT case_number, hearing_year, counsel_for_respondent FROM legal WHERE case_number <= 2 OR case_number >= 17 OR defendant_current_status = 'Business operator' OR case_number = 5;

-- Query 50: 5 (legal)
SELECT plaintiff_current_status, evidence, judge_name FROM legal WHERE plaintiff_current_status != 'Fisherman' OR fine_amount = 20000.00 OR judgment_year != 2009 OR nationality_for_applicant != 'India';

-- Query 51: 6 (legal)
SELECT legal_basis_num, charges, first_judge FROM legal WHERE (charges = 'Unconscionable conduct under s 51AA of the Trade Practices Act 1976' AND counsel_for_applicant = 'Dr J Azzi') OR (nationality_for_applicant != 'Lebanon' AND fine_amount != 210000.00);

-- Query 52: 6 (legal)
SELECT defendant_current_status, charges, judge_name FROM legal WHERE (charges != 'Application for constitutional writs in respect of the Tribunal''s decision' AND case_type != 'Others') OR (first_judge >= 0 AND plaintiff = 'Australian Securities and Investments Commission');

-- Query 53: 6 (legal)
SELECT evidence, verdict, counsel_for_respondent FROM legal WHERE (verdict = 'Others' AND fine_amount = 210000.00) OR (evidence != 1 AND case_number < 2);

-- Query 54: 6 (legal)
SELECT judgment_year, plaintiff, plaintiff_current_status FROM legal WHERE (plaintiff_current_status = 'Former Employee' AND counsel_for_applicant = 'Dr J Renwick') OR (first_judge > 0 AND counsel_for_respondent != '');

-- Query 55: 6 (legal)
SELECT hearing_year, legal_fees, verdict FROM legal WHERE (hearing_year != 2007 AND verdict != 'Guilty') OR (verdict != 'Not Guilty' AND evidence != 1);

-- Query 56: 6 (legal)
SELECT charges, hearing_year, case_type FROM legal WHERE (hearing_year = 2009 AND legal_fees = '3200') OR (first_judge <= 1 AND case_type != 'Others');

-- Query 57: 6 (legal)
SELECT nationality_for_applicant, fine_amount, counsel_for_applicant FROM legal WHERE (fine_amount = 210000.00 AND legal_basis_num < 4) OR (counsel_for_applicant = 'Mr L Karp' AND plaintiff != 'Churche');

-- Query 58: 6 (legal)
SELECT defendant, legal_basis_num, counsel_for_applicant FROM legal WHERE (defendant != 'Minister for Immigration and Multicultural Affairs and Refugee Review Tribunal' AND counsel_for_applicant != 'Mr R Dalton') OR (counsel_for_respondent != 'Ms M Allars' AND hearing_year != 2007);

-- Query 59: 6 (legal)
SELECT fine_amount, plaintiff, verdict FROM legal WHERE (fine_amount = 944120.00 AND evidence > 1) OR (charges != 'Application for constitutional writs in respect of the Tribunal''s decision' AND verdict != 'Not Guilty');

-- Query 60: 6 (legal)
SELECT case_type, case_number, fine_amount FROM legal WHERE (fine_amount != 20000.00 AND first_judge = 1) OR (nationality_for_applicant = 'Philippines' AND first_judge != 1);

