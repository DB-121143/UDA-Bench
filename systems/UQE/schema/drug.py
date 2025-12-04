import json
import os
import pandas as panda

print("Current working directory:", os.getcwd())
data_dir = os.path.join(os.getcwd(), 'data')

class DrugData:
    def __init__(self, path="drug"):
        self.parent_dir = 'data'
        self.dataset_dir = os.path.join(data_dir, 'drug')
        self.source_data_path = os.path.join(self.dataset_dir, 'dataset.json')

        self.dataset_name = 'drug'
        self.tables_name = [
            "drug"
        ]

        drug_schema = {
            "id": "varchar",
            "description": "text",
        }

        drug_columns = [
            "id",
            "description",
        ]

        self.schema = {
            "drug": {
                "columns": drug_columns,
                "schema": drug_schema
            }
        }

        drug_data = None
        with open(self.source_data_path, 'r') as f:
            drug_data = json.load(f)
        self.source_data = {
            "drug": drug_data
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
                info_intro_str += '<' + col_attrs[0] + ": " + self.info_intro[col_name][col_attrs[0]] + ">; "
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
            attr_with_labels_str.append(col_name + '.' + attr_name + ': ' + labels)
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
            "generic_name": "varchar",
            "brand_name": "varchar",
            "disease_name": "varchar",
            "indication": "varchar",
            "active_ingredients": "varchar",
            "pharmaceutical_form": "varchar",
            "manufacturer": "varchar",
            "administration_route": "varchar",
            "recommended_usage": "varchar",
            "single_dose": "varchar",
            "dosage_frequency": "varchar",
            "mechanism_of_action": "varchar",
            "side_effects": "varchar",
            "activation_conditions": "varchar",
            "prescription_status": "varchar",
            "unsuitable_population": "varchar",
            "storage_conditions": "varchar",
        }
        self.columns_with_attr_type = columns_with_attr_type

    def prompt_info_intro_init(self):
        info_intro = {}
        info_intro["description"] = {
            "generic_name": "the standard generic name(s) of the drug (e.g., Sildenafil with Dapoxetine). If multiple generic names, separate them with '||'.",
            "brand_name": "the commercial or brand name of the drug (e.g., Super Kamagra). If multiple brand names, separate them with '||'.",
            "disease_name": "the main disease(s) the drug is intended to treat; use standard disease names (e.g., Hypertension, Diabetes Mellitus, Asthma). If multiple disease names, separate them with '||'.",
            "indication": "the main symptoms, clinical manifestations, or related conditions for which the drug may be used (e.g., fever, pain, edema, high cholesterol, wheezing). If multiple indications, separate them with '||'.",
            "active_ingredients": "active pharmaceutical ingredients in the drug (e.g., Sildenafil||Dapoxetine). If multiple active ingredients, separate them with '||'.",
            "pharmaceutical_form": "the physical form in which the drug is administered; choose one or more from ['tablet', 'capsule', 'injection', 'solution', 'cream', 'ointment', 'powder', 'granule', 'suspension', 'patch', 'spray', 'suppository', 'other']. If multiple pharmaceutical forms, separate them with '||'.",
            "manufacturer": "the name of the company or manufacturer producing the drug (e.g., Ajanta Pharma).",
            "administration_route": "the route through which the drug is administered; choose one or more from ['oral', 'intravenous', 'intramuscular', 'subcutaneous', 'topical', 'transdermal', 'rectal', 'nasal', 'inhalation', 'other']. If multiple administration routes, separate them with '||'.",
            "recommended_usage": "recommended instructions for use, choose one or more from ['swallowed with water', 'chewable', 'dissolve in water', 'take with food', 'take without food', 'take with or without food', 'at fixed time', 'before meals', 'after meals', 'other']. If multiple recommended usages, separate them with '||'.",
            "single_dose": "the amount of the drug to be taken at one time per administration,extract all 'population + dose' pairs (e.g., adults 10 mg || children 5 mg || patients with EGPA 3x100 mg). If multiple single doses, separate them with '||'.",
            "dosage_frequency": "the recommended frequency of administration (e.g., once daily, twice a day, every 8 hours). If multiple dosage frequencies, separate them with '||'.",
            "mechanism_of_action": "how the drug works in the body to achieve its intended effect (e.g., improves blood circulation in penile vessels, supporting erection). If multiple mechanism of actions, separate them with '||'.",
            "side_effects": "known or common side effects associated with the drug (e.g., photosensitivity, nausea, dizziness, drowsiness, tenderness). If multiple side effects, separate them with '||'.",
            "activation_conditions": "specific conditions or requirements for the drug to take effect; choose one or more from ['requires stimulation', 'requires food', 'requires fasting', 'requires physical activity', 'no special condition', 'requires specific timing', 'requires co-administration with other drugs', 'other']. If multiple activation conditions, separate them with '||'.",
            "prescription_status": "whether the drug requires a prescription or can be bought over-the-counter; choose one or more from ['prescription_only', 'over_the_counter', 'restricted', 'unclassified']. If multiple prescription status, separate them with '||'.",
            "unsuitable_population": "main populations for which the drug is not suitable or should be avoided; specify as described in the document (e.g., children, elderly, pregnant women, patients with renal impairment, children under 12). If multiple unsuitable populations, separate them with '||'.",
            "storage_conditions": "recommended conditions for storing the drug; choose one or more from ['store at room temperature', 'store in a cool place', 'refrigerate', 'freeze', 'protect from light', 'keep in a dry place', 'keep tightly closed', 'avoid freezing', 'other']. If multiple storage conditions, separate them with '||'."
        }
        self.info_intro = info_intro
    
    def system_prompt_init(self):
        system_prompt = {}
        system_prompt["description"] = {
            "extract": "You are a medical analyst. The following paragraph " + \
                "is a medical article. *info_intro* " + \
                "Please Help me to extract the value about *attrs_in_prompt*. " + \
                "You should present the result in the form of 'attr: value' and connect different pairs with separator '&&'. " + \
                "For example, 'attr1: 5&&attr2: public&&...'. If the value of the attr is not available, please return 'None'.",
            "filter": "You are a medical analyst. The following paragraph " + \
                "is a medical article. *info_intro* " + \
                "Please Help me to extract the information about *attr_name* " + \
                "and determine whether the it satisfies '*attr_name* *cond_op* *cond_val*'. " + \
                "If it does, please only return True. Otherwise, please return False. Please only return 'True' or 'False'.",
            "groupby_labels": "I will provide you several paragraphs, each paragraph " + \
                "is a medical article. *info_intro* " + \
                "Please help me to group these paragraphs into different labels according to the attribute '*attr_name*'. " + \
                "You should present the labeling result in the form of 'label1||label2||label3||...'. ",
            "groupby_labels_split": "I will provide a medical article. *info_intro* " + \
                "Please help me to extract the attribute '*attr_name*'. " + \
                "You should only present the result, without any other information. ",  
            "merge_filter_column_indicators": "The paragraph 'description' is a medical article in text form.",
            "merge_filter_cond_indicators": "'description': [*col_conditions*]",
        }

        system_prompt["groupby_class"] = \
                "I will provide you several paragraphs with their column names in the following conversation. " + \
                "Also, I will provide a list of attributes, including the possible categories for each attribute." + \
                "Each element of the list will be presented in the form of '<column>.<attribute>:<list of categories>'. " + \
                "The attributes and their candidate catetories are: *attr_with_labels*. " + \
                "Please extract the attribute from the column mentioned in the paragraph and classify them into the given categories. " + \
                "You should present the classification result of each attributes in the form of 'attr1:category1&&attr2:category2&&...&&attrn:categoryn' " + \
                "If none of the given category is matched for the attribute, please return 'attr:None' as its classfication result." + \
                "You should only return the results, without any other information."
        system_prompt["merge_filter"] = \
                "I will provide you several paragraphs in the following conversation. " + \
                "*col_list* " + \
                "The conditions for each paragraph are presented in format '<name1>:[<condition1>,<condition2>,<condition3>,...]'. " + \
                "Each condition is expressed as <attr_name> <cond_op> <cond_val>. " + \
                "Please help me filter the paragraphs based on the given conditions. " + \
                "You should first consider the value of attributes from the paragraph " + \
                "and then determine whether the condition are satisfied. " + \
                "The list of all conditions is: *condition_list*. " + \
                "The explanations for the included attributes are as follows: *info_intro*" + \
                "When an attribute contains multiple values, the operator '=' indicates whether the value is present among those values. " + \
                "For example, if the attribute 'diseases' contains 'breast cancer||lung cancer', and the condition is 'diseases=breast cancer', then the condition is satisfied. " + \
                "You should evaluate all paragraphs against all conditions collectively. " + \
                "Return 'True' if all the conditions are satisfied " + \
                "otherwise, return 'False'." + \
                "Please only return 'True' or 'False', without any additional information."

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
                "and return 'False' if some of the conditions are not satisfied. Please only return 'True' or 'False'."
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
