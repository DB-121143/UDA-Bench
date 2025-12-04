import json
import os
import pandas as panda

print("Current working directory:", os.getcwd())
data_dir = os.path.join(os.getcwd(), 'data')

class ArtImageData:
    def __init__(self, path="Wikiart_image"):
        self.parent_dir = 'data'
        self.dataset_dir = os.path.join(data_dir, path)
        self.source_data_path = os.path.join(self.dataset_dir, 'dataset.json')

        self.dataset_name = 'ArtImage'
        self.tables_name = [
            "Wikiart"
        ]

        art_schema = {
            "id": "varchar",
            "description": "text",
            "image": "image",
        }

        art_columns = [
            "id",
            "description",
            "image",
        ]

        self.schema = {
            "Wikiart": {
                "columns": art_columns,
                "schema": art_schema
            }
        }

        art_data = None
        with open(self.source_data_path, 'r') as f:
            art_data = json.load(f)
        self.source_data = {
            "Wikiart": art_data
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
                col_explain = col_name + "." + col_attr + ": " + self.info_intro[col_name][col_attr]
                column_attrs_list.append(col_explain)
        info_str = '; '.join(column_attrs_list)

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
            "Name": "varchar",
            "Nationality": "varchar",
            "Art_movement": "varchar",
            "Birth_date": "date",
            "Death_date": "date",
            "Age": "int",
            "Century": "varchar",
            "Zodiac": "varchar",
            "Birth_country": "varchar",
            "Birth_city": "varchar",
            "Birth_continent": "varchar",
            "Death_country": "varchar",
            "Death_city": "varchar",
            "Field": "varchar",
            "Genre": "varchar",
            "Marriage": "varchar",
            "Art_institution": "varchar",
            "Teaching": "int",
            "Awards": "int",
        }
        columns_with_attr_type["image"] = {
            "Style": "varchar",
            "Image_genre": "varchar",
            "Object": "varchar",
            "Color": "varchar",
            "Tone": "varchar",
            "Composition": "varchar",
        }
        self.columns_with_attr_type = columns_with_attr_type

    def prompt_info_intro_init(self):
        info_intro = {}
        info_intro["description"] = {
            "Name": "full name of the artist",
            "Nationality": "artist's nationality (e.g., french, japanese). If multiple nationalities, separate them with '||'.",
            "Art_movement": "art movement contributed to (e.g., impressionism). If multiple art movements, separate them with '||'.",
            "Birth_date": "the birth date of the artist in %Y/%-m/%-d format (e.g., 1839/1/9), or leave empty if not applicable",
            "Death_date": "the death date of the artist in %Y/%-m/%-d format (e.g., 1836/1/1), or leave empty if not applicable",
            "Age": "age of the artist (if Death_date is given, compute as Death_date - Birth_date; otherwise, compute as 2025 - Birth_date)",
            "Century": "century when the artist was active or influential (e.g., 19th or 19th-20th; can span multiple centuries; do not include 'century' in the result)",
            "Zodiac": "western zodiac sign based on birth date, choose one form:[Aries, Taurus, Gemini, Cancer, Leo, Virgo, Libra, Scorpio, Sagittarius, Capricorn, Aquarius, Pisces]",
            "Birth_country": "country where the artist was born (e.g., netherlands)",
            "Birth_city": "city where the artist was born",
            "Birth_continent": "continent where the artist was born (e.g., europe, asia)",
            "Death_country": "country where the artist died, or leave empty if not applicable",
            "Death_city": "city where the artist died, or leave empty if not applicable",
            "Field": "primary artistic field, choose one or more from: [Painting, Sculpture, Printmaking, Drawing, Photography, Illustration, Collage, Textile Art, Installation, Performance, Ceramics, Digital Art, Architecture, Mosaic, Graphic Design, Calligraphy]. Multiple fields should be separated by '||'.",
            "Genre": "primary genre of work, choose one or more from: [Portrait, Landscape, Still Life, Religious, Mythological, History, Genre, Nude, Animal, Illustration, Allegorical, Abstract, Fantasy, Sketch, Figurative, Conceptual, Surrealist, Symbolic, Political, Social Realism, Pop Art, Narrative, Environmental, Sculpture, Architecture, Music, Film, Digital Art, Literature, Other]. Multiple genres should be separated by '||'.",
            "Marriage": "marital status of the artist, choose one from: [Cohabiting, Divorced, Married, Remarried, Separated, Unmarried, Widowed]",
            "Art_institution": "Art academies, schools, organizations, associations, or exhibition societies the artist was affiliated with, taught at, or participated in (e.g., national art academies, artist unions, art schools, exhibition societies). If multiple institutions, separate them with '||'.",
            "Teaching": "whether the artist taught at an institution (1 if yes, 0 if none)",
            "Awards": "Number of significant and clearly documented awards the artist received (enter an integer; use 0 if none; exclude medals, scholarships, or honorary mentions).",
        }
        info_intro["image"] = {
            "Style": "The style of the artwork (e.g., realism, abstract).",
            "Image_genre": "The genre of the artwork (e.g., landscape, portrait).",
            "Object": "The primary objects or subjects in the artwork.",
            "Color": "The predominant color used in the artwork.",
            "Tone": "The tonal qualities (e.g., bright, dark, neutral).",
            "Composition": "The composition style of the artwork. Only give composition style, do not give any other information.",
        }

        self.info_intro = info_intro
    
    def system_prompt_init(self):
        system_prompt = {}
        system_prompt["description"] = {
            "extract": "You are an art analyst. The following paragraph " + \
                "provides some basic information about the artist. *info_intro* " + \
                "Please Help me to extract the value about *attrs_in_prompt*. " + \
                "You should present the result in the form of 'attr: value' and connect different pairs with separator '&&'. " + \
                "For example, 'attr1: 5&&attr2: public&&...'. If the value of the attr is not available, please return 'None'.",
            "filter": "You are an art analyst. The following paragraph " + \
                "provides some basic information about the company. *info_intro* " + \
                "Please Help me to extract the information about *attr_name* " + \
                "and determine whether the it satisfies '*attr_name* *cond_op* *cond_val*'. " + \
                "If it does, please only return True. Otherwise, please return False. Please only return 'True' or 'False'.",
            "groupby_labels": "I will provide you several paragraphs, each paragraph " + \
                "is the introdution of an artist. *info_intro* " + \
                "Please help me to group these paragraphs into different labels according to the attribute '*attr_name*'. " + \
                "You should present the labeling result in the form of 'label1||label2||label3||...'. ",  
            "groupby_labels_split": "I will provide an introdution of the artist. *info_intro* " + \
                "Please help me to extract the attribute '*attr_name*'. " + \
                "You should only present the result, without any other information. ",  
            "merge_filter_column_indicators": "The paragraph 'description' is a description of the artist in text form.",
            "merge_filter_cond_indicators": "The conditions on paragraph 'description' include: [*col_conditions*]",
        }

        system_prompt["image"] = {
            "extract": "You are an art analyst. The following image " + \
                "is a work of art. *info_intro* " + \
                "Please Help me to extract the value about *attrs_in_prompt*. " + \
                "You should present the result in the form of 'attr: value' and connect different pairs with separator '&&'. " + \
                "For example, 'attr1: 5&&attr2: public&&...'. If the value of the attr is not available, please return 'None'.",
            "filter": "You are an art analyst. The following image " + \
                "is a work of art. *info_intro* " + \
                "Please Help me to extract the information about *attr_name* " + \
                "and determine whether the it satisfies '*attr_name* *cond_op* *cond_val*'. " + \
                "If it does, please only return True. Otherwise, please return False. Please only return 'True' or 'False'.",
            "merge_filter_column_indicators": "The 'image' is an art work.",
            "merge_filter_cond_indicators": "The conditions on image 'image' include: [*col_conditions*]",
            "groupby_labels": "I will provide you several images, each image " + \
                "is an art work. *info_intro* " + \
                "Please help me to group these images into different labels according to the attribute '*attr_name*'. " + \
                "You should present the labeling result in the form of 'label1||label2||label3||...'. ",  
            "groupby_labels_split": "I will provide an art work. *info_intro* " + \
                "Please help me to extract the attribute '*attr_name*'. " + \
                "You should only present the result, without any other information. ",  
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
                "OR([AND([(<name1>.<attribute1> = value1), (<name2>.<attribute1> < value2)]), AND([(<name1>.<attribute1> = value3)])]). " + \
                "OR means the conditions are ORed. " + \
                "AND means the conditions are ANDed. " + \
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