-- Query 1: 1 (Finan_finance)
SELECT business_segments_num, dividend_per_share, auditor FROM Finan_finance WHERE auditor = 'PKF Littlejohn LLP';

-- Query 2: 1 (Finan_finance)
SELECT net_assets, board_members, net_profit_or_loss FROM Finan_finance WHERE net_profit_or_loss = '1460000000';

-- Query 3: 1 (Finan_finance)
SELECT bussiness_cost, remuneration_policy, registered_office FROM Finan_finance WHERE bussiness_cost != '2129327000';

-- Query 4: 1 (Finan_finance)
SELECT net_profit_or_loss, exchange_code, total_Debt FROM Finan_finance WHERE total_Debt = '21325000000';

-- Query 5: 1 (Finan_finance)
SELECT principal_activities, bussiness_profit, major_events FROM Finan_finance WHERE bussiness_profit = '230400000';

-- Query 6: 1 (Finan_finance)
SELECT major_equity_changes, exchange_code, the_highest_ownership_stake FROM Finan_finance WHERE exchange_code = 'RFT';

-- Query 7: 1 (Finan_finance)
SELECT bussiness_profit, auditor, total_assets FROM Finan_finance WHERE auditor != 'PKF Littlejohn LLP';

-- Query 8: 1 (Finan_finance)
SELECT bussiness_sales, net_profit_or_loss, net_assets FROM Finan_finance WHERE net_assets = 249398000;

-- Query 9: 1 (Finan_finance)
SELECT principal_activities, total_Debt, major_equity_changes FROM Finan_finance WHERE major_equity_changes != 'Yes';

-- Query 10: 1 (Finan_finance)
SELECT largest_shareholder, company_name, executive_profiles FROM Finan_finance WHERE company_name != 'NuCana plc';

-- Query 11: 2 (Finan_finance)
SELECT largest_shareholder, auditor, registered_office FROM Finan_finance WHERE largest_shareholder != 'Kelly Investments 1 PTY Ltd' AND business_risks = 'Environmental Risk';

-- Query 12: 2 (Finan_finance)
SELECT cash_reserves, bussiness_profit, auditor FROM Finan_finance WHERE auditor = 'PKF Littlejohn LLP' AND major_equity_changes = 'Yes';

-- Query 13: 2 (Finan_finance)
SELECT dividend_per_share, bussiness_profit, total_Debt FROM Finan_finance WHERE dividend_per_share != 0.40 AND earnings_per_share = 0.49;

-- Query 14: 2 (Finan_finance)
SELECT net_profit_or_loss, principal_activities, major_equity_changes FROM Finan_finance WHERE net_profit_or_loss != '2823562' AND earnings_per_share = 0.49;

-- Query 15: 2 (Finan_finance)
SELECT total_assets, bussiness_profit, principal_activities FROM Finan_finance WHERE principal_activities != 'Finance' AND exchange_code != 'AIM';

-- Query 16: 2 (Finan_finance)
SELECT remuneration_policy, net_assets, principal_activities FROM Finan_finance WHERE remuneration_policy != 'Stock Option' AND total_assets != 136955488;

-- Query 17: 2 (Finan_finance)
SELECT board_members, revenue, cash_reserves FROM Finan_finance WHERE board_members != 'Keith Bradley' AND business_segments_num < 1;

-- Query 18: 2 (Finan_finance)
SELECT earnings_per_share, the_highest_ownership_stake, net_profit_or_loss FROM Finan_finance WHERE net_profit_or_loss = '165500000' AND bussiness_cost != '993529';

-- Query 19: 2 (Finan_finance)
SELECT largest_shareholder, auditor, major_equity_changes FROM Finan_finance WHERE largest_shareholder != 'Drake Private Investments' AND business_risks != 'Credit Risk';

-- Query 20: 2 (Finan_finance)
SELECT earnings_per_share, cash_reserves, principal_activities FROM Finan_finance WHERE principal_activities = 'Agriculture' AND the_highest_ownership_stake != 11.63;

-- Query 21: 3 (Finan_finance)
SELECT total_assets, board_members, bussiness_cost FROM Finan_finance WHERE bussiness_cost = '10978000000' OR total_Debt = '168459000';

-- Query 22: 3 (Finan_finance)
SELECT the_highest_ownership_stake, net_profit_or_loss, largest_shareholder FROM Finan_finance WHERE the_highest_ownership_stake != 49.28 OR remuneration_policy != 'Fixed';

-- Query 23: 3 (Finan_finance)
SELECT auditor, board_members, major_events FROM Finan_finance WHERE major_events = 'Other' OR principal_activities != 'Biopharmaceuticals';

-- Query 24: 3 (Finan_finance)
SELECT exchange_code, executive_profiles, principal_activities FROM Finan_finance WHERE exchange_code = 'MTA' OR net_assets != 41673000;

-- Query 25: 3 (Finan_finance)
SELECT major_events, dividend_per_share, earnings_per_share FROM Finan_finance WHERE dividend_per_share != 0.00 OR major_equity_changes > 'Yes';

-- Query 26: 3 (Finan_finance)
SELECT remuneration_policy, bussiness_sales, revenue FROM Finan_finance WHERE remuneration_policy != 'Fixed' OR total_Debt != '0';

-- Query 27: 3 (Finan_finance)
SELECT bussiness_profit, largest_shareholder, net_profit_or_loss FROM Finan_finance WHERE bussiness_profit != '122600000' OR cash_reserves = 476657130;

-- Query 28: 3 (Finan_finance)
SELECT net_assets, exchange_code, remuneration_policy FROM Finan_finance WHERE exchange_code = 'ASE' OR company_name != 'Playtech plc';

-- Query 29: 3 (Finan_finance)
SELECT net_profit_or_loss, earnings_per_share, net_assets FROM Finan_finance WHERE net_profit_or_loss != '53392000' OR major_events = 'Major Contract';

-- Query 30: 3 (Finan_finance)
SELECT net_assets, business_risks, auditor FROM Finan_finance WHERE auditor = 'KPMG' OR bussiness_cost = '31755000';

-- Query 31: 4 (Finan_finance)
SELECT business_risks, largest_shareholder, dividend_per_share FROM Finan_finance WHERE dividend_per_share = 0.40 AND major_equity_changes < 'No' AND earnings_per_share = 0.49 AND net_assets = 1362019954;

-- Query 32: 4 (Finan_finance)
SELECT registered_office, bussiness_sales, major_equity_changes FROM Finan_finance WHERE bussiness_sales = '1793368000' AND executive_profiles != 'Michael Boggs' AND revenue != 12857200000 AND business_risks != 'Credit Risk';

-- Query 33: 4 (Finan_finance)
SELECT net_profit_or_loss, business_risks, business_segments_num FROM Finan_finance WHERE business_risks = 'Strategic Risk' AND business_risks != 'Legal/Compliance Risk' AND earnings_per_share != 0.49 AND cash_reserves != 1987600000;

-- Query 34: 4 (Finan_finance)
SELECT earnings_per_share, business_risks, net_assets FROM Finan_finance WHERE business_risks = 'Operational Risk' AND business_segments_num < 2 AND bussiness_sales = '2236000000' AND total_Debt = '3254939000';

-- Query 35: 4 (Finan_finance)
SELECT exchange_code, principal_activities, earnings_per_share FROM Finan_finance WHERE principal_activities != 'Media' AND major_equity_changes != 'Yes' AND net_profit_or_loss = '1723000000' AND earnings_per_share = 0.49;

-- Query 36: 4 (Finan_finance)
SELECT executive_profiles, total_assets, auditor FROM Finan_finance WHERE executive_profiles != 'Nancy L. Erba' AND business_risks != 'Market Risk' AND remuneration_policy = 'Mixed' AND business_segments_num = 2;

-- Query 37: 4 (Finan_finance)
SELECT bussiness_sales, auditor, revenue FROM Finan_finance WHERE bussiness_sales != '-171254' AND business_risks != 'Other' AND the_highest_ownership_stake != 17.70 AND net_assets != 249398000;

-- Query 38: 4 (Finan_finance)
SELECT company_name, business_segments_num, exchange_code FROM Finan_finance WHERE company_name != 'Swiss Water Decaffeinated Coffee Inc.' AND executive_profiles != 'Shayne Currie' AND the_highest_ownership_stake != 49.28 AND total_Debt = '0';

-- Query 39: 4 (Finan_finance)
SELECT total_assets, net_profit_or_loss, bussiness_profit FROM Finan_finance WHERE bussiness_profit = '470734000' AND registered_office != '225 NE Mizner Blvd, Suite 640, Boca Raton, FL 33432' AND largest_shareholder = 'The Master Trust Bank 
of Japan, Ltd. (Trust 
account)' AND net_profit_or_loss = '27299000';

-- Query 40: 4 (Finan_finance)
SELECT total_Debt, bussiness_sales, business_segments_num FROM Finan_finance WHERE total_Debt != '258804000' AND registered_office != '340 Madison Avenue, Suite 3C, New York, New York, 10173' AND major_events != 'Other' AND net_profit_or_loss = '40423000';

-- Query 41: 5 (Finan_finance)
SELECT earnings_per_share, net_profit_or_loss, net_assets FROM Finan_finance WHERE net_profit_or_loss != '165500000' OR principal_activities = 'Other' OR major_equity_changes > 'Yes' OR total_Debt = '8613673';

-- Query 42: 5 (Finan_finance)
SELECT auditor, revenue, remuneration_policy FROM Finan_finance WHERE revenue = 391926000 OR board_members != 'Jakob Pfaudler' OR company_name = 'Bionano Genomics, Inc.' OR auditor != 'BDO LLP';

-- Query 43: 5 (Finan_finance)
SELECT bussiness_sales, registered_office, remuneration_policy FROM Finan_finance WHERE bussiness_sales != '0' OR total_assets != 1358991000 OR board_members = 'Anna Massion' OR total_assets = 33746800000;

-- Query 44: 5 (Finan_finance)
SELECT the_highest_ownership_stake, revenue, board_members FROM Finan_finance WHERE revenue != 12857200000 OR total_assets = 33746800000 OR business_segments_num = 3 OR business_risks = 'Environmental Risk';

-- Query 45: 5 (Finan_finance)
SELECT remuneration_policy, net_profit_or_loss, business_risks FROM Finan_finance WHERE net_profit_or_loss = '27299000' OR the_highest_ownership_stake != 17.70 OR company_name != 'Odyssey Gold Limited' OR earnings_per_share != -0.04;

-- Query 46: 5 (Finan_finance)
SELECT net_assets, executive_profiles, business_risks FROM Finan_finance WHERE net_assets = 249398000 OR net_assets = 1362019954 OR cash_reserves = 87043000 OR total_Debt = '119701564920';

-- Query 47: 5 (Finan_finance)
SELECT dividend_per_share, company_name, executive_profiles FROM Finan_finance WHERE dividend_per_share = 1.12 OR bussiness_cost != '682929600' OR bussiness_profit = '27299000' OR earnings_per_share = 2.18;

-- Query 48: 5 (Finan_finance)
SELECT bussiness_profit, principal_activities, major_events FROM Finan_finance WHERE principal_activities != 'Transportation' OR registered_office != '2 Prologis Blvd., Suite 500, Mississauga, Ontario, L5W 0G8' OR dividend_per_share != 0.40 OR major_events = 'Leadership Change';

-- Query 49: 5 (Finan_finance)
SELECT net_assets, total_assets, auditor FROM Finan_finance WHERE total_assets != 136955488 OR total_assets != 33746800000 OR remuneration_policy != 'Stock Option' OR major_events != 'Major Contract';

-- Query 50: 5 (Finan_finance)
SELECT company_name, the_highest_ownership_stake, business_segments_num FROM Finan_finance WHERE business_segments_num = 1 OR earnings_per_share = -0.04 OR exchange_code = 'WHLR' OR dividend_per_share = 0.40;

-- Query 51: 6 (Finan_finance)
SELECT remuneration_policy, total_Debt, executive_profiles FROM Finan_finance WHERE (executive_profiles = 'Jakob Pfaudler' AND cash_reserves = 87043000) OR (registered_office != 'Beechfield, Hollinhurst Road, Radcliffe, Manchester, M26 1JN' AND auditor != 'BDO Audit Pty Ltd');

-- Query 52: 6 (Finan_finance)
SELECT executive_profiles, business_risks, auditor FROM Finan_finance WHERE (auditor = 'PricewaterhouseCoopers LLP' AND major_equity_changes = 'No') OR (business_risks != 'Legal/Compliance Risk' AND bussiness_sales = '0');

-- Query 53: 6 (Finan_finance)
SELECT major_events, net_profit_or_loss, net_assets FROM Finan_finance WHERE (net_profit_or_loss = '44060000' AND dividend_per_share = 0.40) OR (major_events = 'Major Contract' AND bussiness_cost = '2795073000');

-- Query 54: 6 (Finan_finance)
SELECT board_members, registered_office, the_highest_ownership_stake FROM Finan_finance WHERE (the_highest_ownership_stake = 49.28 AND executive_profiles != 'D N Fletcher') OR (registered_office = '169 Inverness Dr W, Suite 300, Englewood, Colorado 80112' AND bussiness_profit = '1925000000');

-- Query 55: 6 (Finan_finance)
SELECT principal_activities, remuneration_policy, business_segments_num FROM Finan_finance WHERE (principal_activities = 'Real Estate' AND business_risks != 'Other') OR (net_profit_or_loss != '1658085550' AND major_events != 'Leadership Change');

-- Query 56: 6 (Finan_finance)
SELECT executive_profiles, principal_activities, revenue FROM Finan_finance WHERE (principal_activities != 'Agriculture' AND board_members = 'Nigel Machin') OR (bussiness_sales = '176940000' AND exchange_code != 'NYSE');

-- Query 57: 6 (Finan_finance)
SELECT dividend_per_share, earnings_per_share, major_events FROM Finan_finance WHERE (dividend_per_share = 0.40 AND total_Debt != '119701564920') OR (company_name = 'Rectifier Technologies Ltd' AND registered_office != '5 Hanover Quay, Grand Canal Dock, Dublin, D02 VY79, Ireland');

-- Query 58: 6 (Finan_finance)
SELECT exchange_code, bussiness_profit, business_segments_num FROM Finan_finance WHERE (exchange_code != 'MTA' AND cash_reserves != 87043000) OR (total_Debt != '0' AND bussiness_sales != '42963764');

-- Query 59: 6 (Finan_finance)
SELECT dividend_per_share, bussiness_profit, registered_office FROM Finan_finance WHERE (bussiness_profit = '1102000000' AND business_risks != 'Legal/Compliance Risk') OR (bussiness_sales != '42984000' AND net_profit_or_loss != '1745099230');

-- Query 60: 6 (Finan_finance)
SELECT business_segments_num, dividend_per_share, total_assets FROM Finan_finance WHERE (dividend_per_share != 0.00 AND bussiness_sales != '5268700000') OR (auditor = 'BDO LLP' AND bussiness_sales = '10927947660');

