import json
import os
import pandas as panda

print("Current working directory:", os.getcwd())
data_dir = os.path.join(os.getcwd(), 'data')

class InstitutesData:
    def __init__(self, path="institutes"):
        self.parent_dir = 'data'
        self.dataset_dir = os.path.join(data_dir, 'institutes')
        self.source_data_path = os.path.join(self.dataset_dir, 'dataset.json')

        self.dataset_name = 'institutes'
        self.tables_name = [
            "institutes"
        ]

        institute_schema = {
            "id": "varchar",
            "description": "text",
        }

        institute_columns = [
            "id",
            "description",
        ]

        self.schema = {
            "institutes": {
                "columns": institute_columns,
                "schema": institute_schema
            }
        }

        institute_data = None
        with open(self.source_data_path, 'r') as f:
            institute_data = json.load(f)
        self.source_data = {
            "institutes": institute_data
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
            "institution_name": "varchar",
            "institution_type": "varchar",
            "parent_organization": "varchar",
            "establishment_year": "varchar",
            "number_of_staff": "varchar",
            "leadership": "varchar",
            "institution_country": "varchar",
            "institution_city": "varchar",
            "research_diseases": "varchar",
            "research_fields": "varchar",
            "key_technologies": "varchar",
            "key_achievements": "varchar",
            "international_collaboration": "varchar",
            "funding_sources": "varchar",
            "technology_application": "varchar",
        }
        self.columns_with_attr_type = columns_with_attr_type

    def prompt_info_intro_init(self):
        info_intro = {}
        info_intro["description"] = {
            "institution_name": "full name of the medical research institution (e.g., National Cancer Institute, RIKEN SPring-8 Center).",
            "institution_type": "list of the type of institution, choose one from: ['public', 'private', 'university-affiliated',  'corporate research lab'].",
            "parent_organization": "parent or affiliated organization, if any (e.g., Harvard University, Fudan University).",
            "establishment_year": "the year the institution was established (e.g., 1996).",
            "number_of_staff": "number of staff or researchers in the institution (e.g., 250).",
            "leadership": "leading figures or directors, e.g., name of the head scientist, director, or notable experts.",
            "institution_country": "the country where the institution is located (e.g., USA, Japan, China, Germany).",
            "institution_city": "the city where the institution is located (e.g., Boston, Tokyo, Beijing, Berlin).",
            "research_diseases": "main diseases that the institution focuses its research on (e.g., breast cancer, diabetes, Alzheimer's disease, inflammatory bowel disease). If multiple diseases, separate them with '||'.",
            "research_fields": "main areas or directions of medical research the institution focuses on, choose one or more from: ['molecular_biology', 'regenerative_medicine', 'immunology', 'genomics', 'pharmacology', 'oncology', 'neuroscience', 'bioinformatics', 'microbiology', 'epidemiology', 'clinical_research', 'biochemistry', 'cell_biology', 'virology', 'pathology', 'public_health', 'medical_imaging', 'others']. If multiple fields, separate them with '||'.",
            "key_technologies": "key technologies, methods, or platforms used by the institution (e.g., automated specimen tracking, X-ray crystallography, CAR-T cell therapy). If multiple technologies, separate them with '||'.",
            "key_achievements": "major research achievements, scientific breakthroughs, or certifications obtained by the institution (e.g., ISO15189 certification, protein structure determination, discovery in cardiometabolic diseases). If multiple achievements, separate them with '||'.",
            "international_collaboration": "partnerships or collaborations with other institutions or companies, including joint projects and investments (e.g., collaboration between IPC and AP-HM, pharma investment in CAR-T therapy). If multiple collaborations, separate them with '||'.",
            "funding_sources": "main sources of funding or grants supporting the institution's research, choose one or more from: ['government grant', 'industry grant', 'philanthropy', 'self funded', 'international grant', 'other']. If multiple sources, separate them with '||'.",
            "technology_application": "practical applications or implementations of the institution's technologies in clinical or research settings (e.g., barcode specimen handling, therapy for lipodystrophy, omega-3 supplements for heart health). If multiple applications, separate them with '||'."
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
                "Please help me filter the paragraphs according to the given conditions. " + \
                "The conditions on a paragraph are presented in format '<name1>:[<condition1>,<condition2>,<condition3>,...]'. " + \
                "Each condition is expressed as <attr_name> <cond_op> <cond_val>. " + \
                "You should first consider the value of <attr_name> from the paragraph " + \
                "and then determine whether the condition is satisfied. " + \
                "These conditions are: *condition_list*. " + \
                "The explanations for the included attributes are as follows: *info_intro*" + \
                "When an attribute contains multiple values, the operator '=' indicates whether the value is present among those values. " + \
                "Return 'True' if all the conditions are satisfied " + \
                "and return 'False' otherwise. " + \
                "Please only return the 'True' or 'False', without other information."
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
