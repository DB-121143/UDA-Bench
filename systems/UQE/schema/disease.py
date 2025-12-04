import json
import os
import pandas as panda

print("Current working directory:", os.getcwd())
data_dir = os.path.join(os.getcwd(), 'data')

class DiseaseData:
    def __init__(self, path="disease"):
        self.parent_dir = 'data'
        self.dataset_dir = os.path.join(data_dir, 'disease')
        self.source_data_path = os.path.join(self.dataset_dir, 'dataset.json')

        self.dataset_name = 'disease'
        self.tables_name = [
            "disease"
        ]

        disease_schema = {
            "id": "varchar",
            "description": "text",
        }

        disease_columns = [
            "id",
            "description",
        ]

        self.schema = {
            "disease": {
                "columns": disease_columns,
                "schema": disease_schema
            }
        }

        disease_data = None
        with open(self.source_data_path, 'r') as f:
            disease_data = json.load(f)
        self.source_data = {
            "disease": disease_data
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
            "disease_name": "varchar",
            "disease_type": "varchar",
            "pathogenesis": "varchar",
            "etiology": "varchar",
            "diagnostic_methods": "varchar",
            "common_symptoms": "varchar",
            "complications": "varchar",
            "affected_organs": "varchar",
            "treatments": "varchar",
            "drugs": "varchar",
            "prognosis": "varchar",
            "sequelae": "varchar",
            "epidemiology": "varchar",
            "risk_factors": "varchar",
            "preventive_measures": "varchar",
            "diagnosis_challenges": "varchar",
            "treatment_challenges": "varchar",
            "quality_of_life_impact": "varchar"
        }
        self.columns_with_attr_type = columns_with_attr_type

    def prompt_info_intro_init(self):
        info_intro = {}
        info_intro["description"] = {
            "disease_name": "the official name of the disease (e.g., Rheumatoid Arthritis).",
            "disease_type": "comprehensive classification of the disease; choose one or more from: ['autoimmune', 'infectious', 'genetic', 'metabolic', 'neoplastic', 'degenerative', 'congenital', 'traumatic', 'nutritional', 'vascular', 'inflammatory', 'endocrine', 'psychiatric', 'toxic', 'allergic', 'iatrogenic', 'idiopathic', 'parasitic', 'others']. If multiple types, separate them with '||'.",
            "pathogenesis": "the biological and physiological process by which the disease develops and progresses, choose one or more from: ['autoimmune', 'infectious_bacterial', 'infectious_viral', 'infectious_fungal', 'infectious_parasitic', 'genetic_mutation', 'metabolic_disorder', 'inflammatory', 'degenerative', 'neoplastic', 'vascular', 'traumatic', 'toxic', 'allergic', 'endocrine_disorder', 'nutritional_deficiency', 'iatrogenic', 'idiopathic', 'other']. If multiple pathogenesis, separate them with '||'.",
            "etiology": "the underlying cause(s) of the disease, including genetic, environmental, infectious, or lifestyle factors (e.g., gene mutation, viral infection, smoking). If multiple etiologies, separate them with '||'.",
            "diagnostic_methods": "primary diagnostic approaches used for this disease; choose one or more from: ['clinical_evaluation', 'laboratory_test', 'imaging', 'biopsy', 'genetic_testing', 'endoscopy', 'functional_test', 'pathology', 'screening', 'other']. If multiple diagnostic methods, separate them with '||'.",
            "common_symptoms": "typical clinical manifestations or symptoms experienced by patients with this disease (e.g., fever, cough, joint pain). If multiple symptoms, separate them with '||'.",
            "complications": "significant secondary conditions or health problems that can arise as a result of the disease (e.g., heart failure, kidney injury). If multiple complications, separate them with '||'.",
            "affected_organs": "primary organs, tissues, or body systems impacted by the disease (e.g., lungs, liver, joints, nervous system). If multiple organs, separate them with '||'.",
            "treatments": "list of treatment modalities, choose one or more from: ['oral_medication', 'injection', 'surgery', 'radiotherapy', 'chemotherapy', 'physical_therapy', 'psychotherapy', 'lifestyle_modification', 'device_implantation', 'dietary_intervention', 'others']. If multiple treatments, separate them with '||'.",
            "drugs": "list the specific names of drugs or pharmaceuticals used to treat or control the disease (e.g., metformin, aspirin, infliximab, penicillin). If multiple drugs, separate them with '||'.",
            "prognosis": "typical course and expected outcome of the disease; choose one or more from: ['full_recovery', 'partial_recovery', 'chronic_condition', 'remission_possible', 'high_recurrence_risk', 'low_recurrence_risk', 'high_mortality', 'moderate_mortality', 'low_mortality', 'disability', 'variable', 'stable', 'progressive', 'quality_of_life_impairment', 'asymptomatic', 'unknown']. If multiple prognoses, separate them with '||'.",
            "sequelae": "long-term or permanent health problems, symptoms, or functional impairments that persist after the acute phase of a disease or injury has resolved.(e.g., paralysis after stroke, chronic pain, hearing loss or cognitive impairment). If multiple sequelae, separate them with '||'.",
            "epidemiology": "population-level data about the disease, including prevalence, incidence, age/gender distribution, or notable epidemiological patterns (e.g., 1 in 1000 adults, more common in males).",
            "risk_factors": "factors that increase the likelihood of developing the disease, choose one or more from: ['family_history', 'genetic_predisposition', 'age', 'gender', 'smoking', 'alcohol_consumption', 'obesity', 'overweight', 'sedentary_lifestyle', 'unhealthy_diet', 'hypertension', 'hyperlipidemia', 'diabetes', 'chronic_disease', 'immunosuppression', 'environmental_exposure', 'occupational_exposure', 'exposure_to_toxins', 'infection', 'stress', 'low_socioeconomic_status', 'radiation_exposure', 'other']. If multiple risk factors, separate them with '||'.",
            "preventive_measures": "known methods or actions to reduce the risk of disease occurrence, choose one or more from: ['vaccination', 'lifestyle_modification', 'screening', 'prophylactic_medication', 'personal_hygiene', 'health_education', 'protective_equipment', 'environmental_control', 'vector_control', 'safe_sex_practices', 'regular_exercise', 'dietary_management', 'early_detection', 'other']. If multiple preventive measures, separate them with '||'.",
            "diagnosis_challenges": "typical difficulties or obstacles encountered in the diagnosis of the disease (e.g., nonspecific symptoms, lack of early biomarkers, disease mimics). If multiple challenges, separate them with '||'.",
            "treatment_challenges": "major issues in the treatment or management of the disease (e.g., drug resistance, frequent relapse, side effects, poor adherence). If multiple challenges, separate them with '||'.",
            "quality_of_life_impact": "how the disease affects the patient's daily living, social interactions, work, and psychological well-being choose one or more from: ['mobility_impairment', 'chronic_pain', 'fatigue', 'sleep_disorders', 'anxiety', 'depression', 'social_isolation', 'work_disability', 'cognitive_impairment', 'sexual_dysfunction', 'self_care_difficulty', 'dietary_restriction', 'dependence_on_others', 'body_image_issues', 'financial_burden', 'emotional_distress', 'stigma', 'quality_of_life_decline', 'other']. If multiple impacts, separate them with '||'."
        }
        self.info_intro = info_intro
    
    def system_prompt_init(self):
        system_prompt = {}
        system_prompt["description"] = {
            "extract": "You are a disease analyst. The following paragraph " + \
                "is a medical article. *info_intro* " + \
                "Please Help me to extract the value about *attrs_in_prompt*. " + \
                "You should present the result in the form of 'attr: value' and connect different pairs with separator '&&'. " + \
                "For example, 'attr1: 5&&attr2: public&&...'. If the value of the attr is not available, please return 'None'.",
            "filter": "You are a disease analyst. The following paragraph " + \
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
            "merge_filter_cond_indicators": "The conditions on paragraph 'description' include: [*col_conditions*]",
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
