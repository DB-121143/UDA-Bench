import json
import os
import pandas as panda

print("Current working directory:", os.getcwd())
data_dir = os.path.join(os.getcwd(), 'data')

class FinRecordData:
    def __init__(self, path="Finance"):
        self.parent_dir = 'data'
        self.dataset_dir = os.path.join(data_dir, path)
        self.source_data_path = os.path.join(self.dataset_dir, 'dataset.json')

        self.dataset_name = 'Finance'
        self.tables_name = [
            "financial"
        ]

        fin_record_schema = {
            "id": "varchar",
            "description": "text",
        }

        fin_record_columns = [
            "id",
            "description",
        ]

        self.schema = {
            "financial": {
                "columns": fin_record_columns,
                "schema": fin_record_schema
            }
        }

        fin_record_data = None
        with open(self.source_data_path, 'r') as f:
            fin_record_data = json.load(f)
        self.source_data = {
            "financial": fin_record_data
        }

        self.columns_with_attr_type_init()
        self.prompt_info_intro_init()
        self.system_prompt_init()

    def get_filter_prompt(self, col_name, col_attr, cond_val, cond_op):
        print("get_filter_prompt...")
        print(col_name, col_attr, cond_val, cond_op)

        info_intro = self.info_intro[col_name][col_attr]
        system_prompt = self.system_prompt[col_name]["filter"].replace("*info_intro*", info_intro).replace("*attr_name*", col_attr).replace("*cond_op*", cond_op).replace("*cond_val*", cond_val)
        return system_prompt
    
    def get_merged_filter_prompt(self, col_to_add):
        print("merged_filter_prompt_gen", col_to_add)
        col_schema_list = []
        condition_list = []
        info_intro_str = ""
        for col_name, col_attrs_list in col_to_add.items():
            col_conditions = []
            for col_attrs in col_attrs_list:
                col_conditions.append(col_attrs[0] + " " + col_attrs[2] + " " + col_attrs[1])
                info_intro_str += col_attrs[0] + ": " + self.info_intro[col_name][col_attrs[0]] + "; "
            col_schema_str = self.system_prompt[col_name]["merge_filter_column_indicators"]
            col_schema_list.append(col_schema_str)
            col_cond_str = self.system_prompt[col_name]["merge_filter_cond_indicators"].replace("*col_conditions*", '; '.join(col_conditions))
            condition_list.append(col_cond_str)
        col_list_str = ' '.join(col_schema_list)
        condition_list_str = '. '.join(condition_list)
        system_prompt = self.system_prompt["merge_filter"].replace("*col_list*", col_list_str).replace("*condition_list*", condition_list_str).replace("*info_intro*", info_intro_str)
        print(system_prompt)

        return system_prompt

    def get_filter_aggr_prompt(self, filter_cond_str, filter_column_attrs):
        print("get_filter_aggr_prompt...")
        column_attrs_list = []
        for col_name, col_attrs_list in filter_column_attrs.items():
            for col_attr in col_attrs_list:
                col_explain = self.info_intro[col_name][col_attr]
                column_attrs_list.append(col_explain)
        info_str = ' '.join(column_attrs_list)

        system_prompt = self.system_prompt["merge_filter_aggr"].replace("*filter_condition*", filter_cond_str).replace("*info_intro*", info_str)
        return system_prompt
    
    def get_filter_aggr_prompt_split(self, and_strings, filter_column_attrs):
        print("get_filter_aggr_prompt_split...")

        system_prompts = []
        for and_string in and_strings:
            column_attrs_list = []
            for col_name, col_attrs_list in filter_column_attrs.items():
                for col_attr in col_attrs_list:
                    if col_attr not in and_string:
                        continue
                    col_explain = self.info_intro[col_name][col_attr]
                    column_attrs_list.append(col_explain)
            info_str = ' '.join(column_attrs_list)

            system_prompt = self.system_prompt["merge_filter_aggr_split"].replace("*filter_condition*", and_string).replace("*info_intro*", info_str)
            system_prompts.append(system_prompt)
        return system_prompts

    def get_extract_prompt(self, col_name, attrs):
        print("get_extract_prompt...")
        print(col_name, attrs)
        attrs_in_prompt = ', '.join(attrs)

        info_intro = ""
        for attr in attrs:
            info_intro = info_intro + col_name + "." + attr + ": " + self.info_intro[col_name][attr] + "; "
        print(info_intro)
        system_prompt = self.system_prompt[col_name]["extract"].replace("*info_intro*", info_intro).replace("*attrs_in_prompt*", attrs_in_prompt)
        return system_prompt
    
    def get_groupby_labels_prompt(self, col_name, attr_name):
        print("get_groupby_labels_prompt...")
        info_intro = col_name + "." + attr_name + ": " + self.info_intro[col_name][attr_name]
        system_prompt = self.system_prompt[col_name]["groupby_labels"].replace("*info_intro*", info_intro).replace("*attr_name*", attr_name)
        return system_prompt
    
    def get_groupby_labels_split_prompt(self, col_name, attr_name):
        print("get_groupby_labels_split_prompt...")
        info_intro = col_name + "." + attr_name + ": " + self.info_intro[col_name][attr_name]
        system_prompt = self.system_prompt[col_name]["groupby_labels_split"].replace("*info_intro*", info_intro).replace("*attr_name*", attr_name)
        return system_prompt
    
    def get_groupby_classify_prompt(self, labels_dict, attr_with_labels):
        print("get_groupby_classify_prompt...")
        attr_with_labels_str = []
        for key, value in labels_dict.items():
            col_name = key.split('.')[0]
            attr_name = key.split('.')[1]
            labels = '[' + ', '.join(value) + ']'
            attr_with_labels_str.append(col_name + '.' + attr_name + ':' + labels)
        attr_with_labels_str = '||'.join(attr_with_labels_str)
        print(attr_with_labels_str)

        system_prompt = self.system_prompt["groupby_class"].replace("*attr_with_labels*", attr_with_labels_str)
        return system_prompt 

    def get_col_type(self, col_name):
        table_name = self.tables_name[0]
        col_name = col_name.split('.')[0]
        return self.schema[table_name]['schema'][col_name]
    
    def get_attr_type(self, col, attr):
        return self.columns_with_attr_type[col][attr]
    
    def get_all_columns(self):
        table_name = self.tables_name[0]
        return self.schema[table_name]["columns"]
    
    def columns_with_attr_type_init(self):
        columns_with_attr_type = {}
        columns_with_attr_type["description"] = {
            "company_name": "varchar",
            "registered_office": "varchar",
            "exchange_code": "varchar",
            "principal_activities": "varchar",
            "board_members": "varchar",
            "executive_profiles": "varchar",
            "revenue": "float",
            "net_profit_or_loss": "float",
            "total_Debt": "float",
            "total_assets": "float",
            "cash_reserves": "float",
            "net_assets": "float",
            "earnings_per_share": "float",
            "dividend_per_share": "float",
            "largest_shareholder": "varchar",
            "the_highest_ownership_stake": "float",
            "major_equity_changes": "varchar",
            "major_events": "varchar",
            "bussiness_sales": "float",
            "bussiness_profit": "float",
            "bussiness_cost": "float",
            "business_segments_num": "int",
            "business_risks": "varchar",
            "remuneration_policy": "varchar",
            "auditor": "varchar"
        }
        self.columns_with_attr_type = columns_with_attr_type

    def prompt_info_intro_init(self):
        info_intro = {}
        info_intro["description"] = {
            "company_name": "the full legal name of the company as disclosed in the annual report.",
            "registered_office": "the registered address or main office location of the company.",
            "exchange_code": "The code of the stock exchange where the company is listed (e.g., ASX ticker symbol).",
            "principal_activities": "main business activities or industry classification; choose one or more from ['Mining', 'Finance', 'Healthcare', 'Manufacturing', 'Technology', 'Retail', 'Energy', 'Utilities', 'Real Estate', 'Transportation', 'Agriculture', 'Telecommunications', 'Media', 'Other']. If multiple choices, separate with '||'.",
            "board_members": "The names of board members. If multiple names, separate with '||'.",
            "executive_profiles": "The names of executive management team members. If multiple names, separate with '||'.",
            "revenue": "total operating revenue recognized during the reporting period. Enter a number. If the original currency is not USD, convert to USD according to the latest available exchange rate.",
            "net_profit_or_loss": "net profit or loss amount for the reporting period (enter a negative number for loss). Enter a number. If the original currency is not USD, convert to USD according to the latest available exchange rate.",
            "total_Debt": "total debt outstanding at the end of the reporting period. Enter a number. If the original currency is not USD, convert to USD according to the latest available exchange rate.",
            "total_assets": "total assets held by the company at the end of the reporting period. Enter a number. If the original currency is not USD, convert to USD according to the latest available exchange rate.",
            "cash_reserves": "total cash and cash equivalents held at period-end. Enter a number. If the original currency is not USD, convert to USD according to the latest available exchange rate.",
            "net_assets": "net asset value at the end of the reporting period. Enter a number. If the original currency is not USD, convert to USD according to the latest available exchange rate.",
            "earnings_per_share": "earnings per share (EPS) for the reporting period. Enter a number. If the original currency is not USD, convert to USD according to the latest available exchange rate.",
            "dividend_per_share": "dividends paid per share for the period. Enter a number. If the original currency is not USD, convert to USD according to the latest available exchange rate.",
            "largest_shareholder": "the name of the largest shareholder.",
            "the_highest_ownership_stake": "the highest percentage of ownership held by a single shareholder. Enter a number, unit: percent (%).",
            "major_equity_changes": "whether there were major equity changes during the period; choose one from ['Yes', 'No'].",
            "major_events": "type(s) of major events; select one or more from ['M&A', 'Litigation', 'Major Contract', 'Leadership Change', 'Restructuring', 'Delisting', 'Other']. If multiple choices, separate with '||'.",
            "bussiness_sales": "total operating revenue from all major business lines or segments. If the document reports revenue for multiple segments separately, sum all segments to obtain the total. Enter a single number. If the original currency is not USD, convert to USD according to the latest available exchange rate.",
            "bussiness_profit": "total profit contributed by all major business lines or projects. If the document reports profit for multiple segments separately, sum all segments to obtain the total. Enter a single number. If the original currency is not USD, convert to USD according to the latest available exchange rate.",
            "bussiness_cost": "total cost associated with all main business lines or projects. If the document reports cost for multiple segments separately, sum all segments to obtain the total. Enter a single number. If the original currency is not USD, convert to USD according to the latest available exchange rate.",
            "business_segments_num": "number of business segments reported.",
            "business_risks": "main types of business risks; select one or more from ['Market Risk', 'Credit Risk', 'Operational Risk', 'Legal/Compliance Risk', 'Environmental Risk', 'Strategic Risk', 'Other']. If multiple choices, separate with '||'.",
            "remuneration_policy": "executive remuneration policy type; select one or more from ['Fixed', 'Performance-based', 'Stock Option', 'Mixed', 'Not Disclosed']. If multiple choices, separate with '||'.",
            "auditor": "name of the external audit firm."
        }
        self.info_intro = info_intro
    
    def system_prompt_init(self):
        system_prompt = {}
        system_prompt["description"] = {
            "extract": "You are an financial analyst. The following paragraph " + \
                "provides some basic information about the company. *info_intro* " + \
                "Please Help me to extract the value about *attrs_in_prompt*. " + \
                "You should present the result in the form of 'attr: value' and connect different pairs with separator '&&'. " + \
                "For example, 'attr1: 5&&attr2: public&&...'. If the value of the attr is not available, please return 'None'.",
            "filter": "You are an financial analyst. The following paragraph " + \
                "provides some basic information about the company. *info_intro* " + \
                "Please Help me to extract the information about *attr_name* " + \
                "and determine whether the it satisfies '*attr_name* *cond_op* *cond_val*'. " + \
                "If it does, please only return True. Otherwise, please return False. Please only return 'True' or 'False'.",
            "groupby_labels": "I will provide you several paragraphs, each paragraph " + \
                "is the introdution of an company. *info_intro* " + \
                "Please help me to group these paragraphs into different labels according to the attribute '*attr_name*'. " + \
                "You should present the labeling result in the form of 'label1||label2||label3||...'. ",  
            "groupby_labels_split": "I will provide an introdution of the company. *info_intro* " + \
                "Please help me to extract the attribute '*attr_name*'. " + \
                "You should only present the result, without any other information. ",  
            "merge_filter_column_indicators": "The paragraph 'description' is a description of the company in text form.",
            "merge_filter_cond_indicators": "The conditions on paragraph 'description' include: [*col_conditions*]",
        }

        system_prompt["groupby_class"] = \
                "I will provide you several paragraphs with their column names in the following conversation. " + \
                "Also, I will provide a list of attributes, including the possible categories for each attribute " + \
                "in the form of '<column>.<attribute>:<list of categories>'. " + \
                "The attributes and their candidate catetories are: *attr_with_labels*. " + \
                "Please extract the attribute from the corresponding paragraph and classify them into the given categories. " + \
                "You should present the classification result of each attributes in the form of 'attr1:category1&&attr2:category2&&...' " + \
                "If none of the given category is matched for the given attribute, please return 'attr:None' as its classfication result." + \
                "You should only return the results, without any other information."
        system_prompt["merge_filter"] = \
                "I will provide you several paragraphs in the following conversation. " + \
                "Each paragraph is preceded by a name indicator. " + \
                "*col_list* " + \
                "Please help me to filter the paragraphs according to the given conditions. " + \
                "The conditions on a paragraph are presented in the form of '<name1>:[<condition1>,<condition2>,<condition3>,...]'. " + \
                "Each condition is presented in the form of <attr_name> <cond_op> <cond_val>. " + \
                "You should first extract the information of <attr_name> from the paragraph " + \
                "and then determine whether the condition is satisfied. " + \
                "The conditions are: *condition_list*. " + \
                "The explaination on the attributes included in the conditions are: *info_intro*" + \
                "You only need to give an overall decision on all these paragraphs and conditions. " + \
                "Return 'True' if all the conditions are satisfied " + \
                "and return 'False' if some of the conditions are not satisfied. " + \
                "Please only return the result 'True' or 'False', without other information."
        system_prompt["merge_filter_aggr"] = \
                "I will provide you several paragraphs in the following conversation. " + \
                "Each paragraph is preceded by a name indicator. " + \
                "Please help me to filter the paragraphs according to the given conditions. " + \
                "The conditions are presented in the specific form with operators, OR and AND. For example: " + \
                "OR([AND([COMP(<name1>.<attribute1> = value1), COMP(<name2>.<attribute1> < value2)]), AND([COMP(<name1>.<attribute1> = value3)])]). " + \
                "OR means the conditions are ORed. " + \
                "AND means the conditions are ANDed. " + \
                "COMP means the condition is a comparison. " + \
                "The name is the name of the paragraph, and the attribute is the attribute of the paragraph. " + \
                "The explaination on the attributes included in the conditions are: *info_intro*" + \
                "The condition to be satisfied is: *filter_condition* " + \
                "Please help me to filter the paragraphs according to the given condition. " + \
                "Return 'True' if all the conditions are satisfied " + \
                "and return 'False' if some of the conditions are not satisfied. Please Only return 'True' or 'False'."
        system_prompt["merge_filter_aggr_split"] = \
                "I will provide you several paragraphs in the following conversation. " + \
                "Each paragraph is preceded by a name indicator. " + \
                "Please help me to filter the paragraphs according to the given condition. " + \
                "The conditions to be satisfied are: *filter_condition* " + \
                "Each condition is presented in the form of (<name1>.<attribute1> <op> <value1>). " + \
                "The explaination on the attributes included in the conditions are: *info_intro*" + \
                "Return 'True' if all the conditions are satisfied " + \
                "and return 'False' otherwise. Please Only return 'True' or 'False', without any other information."
        
        self.system_prompt = system_prompt