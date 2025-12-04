import re
import pandas as pd
from litellm import completion, batch_completion
from tqdm import tqdm
import tiktoken
import os
import logging
import quest.conf.settings as settings
from quest.utils import table_util
from quest.utils.log import print_log
import copy

def parse_result(text, doc_id, attributeList):
    #dic = dict(re.findall(r"(\w+):\s*(.*)", text))
    dic = dict(re.findall(r"(\w+(?:\.\w+)?):([^:\n]*)", text))
    dic["doc_id"] = doc_id
    for attr in attributeList:
        dic.setdefault(attr, None)
    return dic

class LLMInfo(object):
    # static variables
    tot_query_times = 0
    tot_input_tokens = 0
    tot_output_tokens = 0

    @staticmethod
    def add_query_times(time):
        LLMInfo.tot_query_times += time
    
    @staticmethod
    def add_input_tokens(tokens):
        LLMInfo.tot_input_tokens += tokens

    @staticmethod
    def add_output_tokens(tokens):
        LLMInfo.tot_output_tokens += tokens

class TextLLMQuerier(object):
    """
    used for extract attributes with LLMs
    """
    def __init__(self, prompt, llm=settings.LLM_MODEL, api_base=settings.API_BASE):
        self.api_base= api_base
        self.llm = llm
        self.attr_descriptions = prompt
        self.parse_attr_descriptions()
    
    def parse_attr_descriptions(self):
        # 已知初始的attr_descriptions格式如下：
        # 1 按行分隔，每行对应1个attr和对应的description
        # 2 每行的格式为：attr_name: description
        # 要求得到self.attr_descriptions_dict
        self.attr_descriptions_dict = {}
        descriptions = self.attr_descriptions.split("\n")
        for line in descriptions:
            if line.strip() == "":
                continue
            attr_name, description = line.split(":")
            self.attr_descriptions_dict[attr_name.strip()] = description.strip()
        return
    
    def build_text_list(self, textDict):
        # {doc_id1 : { column1 :[(text1, chunkid1), (text2,chunkid2), ...], } }
        doc_idList = list(textDict.keys())
        textList = [] # textList is a list of texts for the column

        for doc_id, columns in textDict.items():
            # columns is a dict, each key is a column name, and the value is a list of texts
            now_text = ""
            chunkid_set = set()
            cnt = 0

            for column, chunkList in columns.items():
                for chunk in chunkList:
                    chunkid = chunk[1]
                    if chunkid in chunkid_set:
                        continue
                    
                    chunkid_set.add(chunkid)
                    cnt += 1
                    now_text = now_text + f'''<Chunk {cnt} begin>\n\n''' +  str(chunk[0]) + f'''\n\n<Chunk {cnt}end>\n\n'''
                
            textList.append(now_text)
        
        #print("textList:\n", textList)
        return textList, doc_idList

    def extract_attribute_from_textDict(self, textDict, attributeList):
        # {doc_id1 : { column1 :[(text1, chunkid1), (text2,chunkid2), ...], } }
        textList, doc_idList = self.build_text_list(textDict)
        # print_log("textList:\n", textList, "\ndoc_idList:\n", doc_idList)
        return self.extract_attribute(textList, doc_idList, attributeList)
    
    def extract_attribute_from_textDict_semantic_fiter(self, textDict, attributeList, filterList):
        # {doc_id1 : { column1 :[(text1, chunkid1), (text2,chunkid2), ...], } }
        textList, doc_idList = self.build_text_list(textDict)
        return self.extract_attribute_and_semantic_filter(textList, doc_idList, attributeList, filterList)

    def extract_attribute_and_semantic_filter(self, textList, doc_idList, attributeList, filterList):
        """
        textList : list[str], each element is a text document, corresponding to doc_idList
        doc_idList : list[str], id of the text documents, corresponding to textList
        attributeList : list[str], attributes to extract from the text documents, form like age or Player.age !!! always input only one attribute
        filterList : list[str], filter conditions to apply on the extracted attributes

        extract the attributes from the textList, which is a list of text documents.

        output : a dataframe, columns = attributeList + ['doc_id'] + ['fcondition']
        """
        docs = copy.copy(textList)

        """
        for file in textList:
            tokens = settings.enc.encode(file)
            truncated_tokens = tokens[:4000]
            truncated_text = settings.enc.decode(truncated_tokens)
            docs.append(truncated_text)
        """

        attributes = ", ".join(attributeList)
        filters = ", ".join(filterList)
        
        related_attr_descriptions = []
        for attr in attributeList:
            related_attr_descriptions.append(f"{attr}: {self.attr_descriptions_dict.get(attr)}")
        related_attr_descriptions_str = " \n".join(related_attr_descriptions)
        prompts = [
            [
                {"role": "system", "content": "You are an information extraction and check assistant. Respond in two lines, the first line is a key-value pair using the exact field name provided; the second line is a key-value pair, and value is a boolean True or False whether the condition is met. Do not include any explanations or extra text."},
                {"role": "user", "content": f'''Extract the following field from the given document: {attributes}. Then Check if the value satisfy the condition.

                Instructions:
                - Format your response as two lines, the first line in the format: `field: value`, and the second line in the format: `fcondition: True/False`.
                - Use the exact field name: {attributes}.
                - Check the condition {filters}.
                - If the field is missing or unknown, leave its value empty (e.g., `team: None`), and leave the condition as False.
                - use the line break (`\\n`) to split the lines.
                - You should first extract the field value and then check. For example, we first do extract, and get filed is \'name\', value is \'Lee\'. Then we do check, the condition is \'name==\'Frank\' \', the value does not satisfy the condition, so the fcondtion is False.
                - The filter condition `==` or `IN` can be considered emantically for strings. For example, \'Lakers\' and \'Los Angeles Lakers\' are equal, \'fashion\' and \'Fashion || Illustration\' are also equal.
                - The filter conditioin `<`, `>`, `>=`, `<=` can be considered as numeric comparison or a date comparison, note that the eariler date is smaller.
                - For example, the filed is \'birth date\', and value is \'2001/10/6\'. Then we do check, the condition is \'birth date<\'1999/11/6\' \', the value does not satisfy the condition, so the fcondition is False, output fcondition: False.
                - For example, the filed is \'style\', and value is \'fashion\'. Then we do check, the condition is \'style == Fashion\', the value satisfy the condition, so the fcondition is True, output fcondition: True.
                - Follow the descriptions of the field:
                ``` {related_attr_descriptions_str} ```
                - Do not add any extra text, comments, quotes, or explanations.

                Document:
                {doc}
                '''
                }
            ]
            for doc in docs
        ]
        results = self.batch_llm_response(prompts=prompts) # totest

        attributeList.append('fcondition')
        json_result = [parse_result(results[i], doc_idList[i], attributeList) for i in range(len(results))]
        df = pd.DataFrame(json_result)
        df = df.fillna(" ")
        attributeList.append('doc_id')
        df = table_util.check_missing_columns(df, attributeList)
        #print("use prompt:", related_attr_descriptions_str)
        print("------------\n", df)
        return df
    
    def extract_attribute(self, textList, doc_idList, attributeList):
        """
        textList : list[str], each element is a text document, corresponding to doc_idList
        doc_idList : list[str], id of the text documents, corresponding to textList
        attributeList : list[str], attributes to extract from the text documents, form like age or Player.age

        extract the attributes from the textList, which is a list of text documents.

        output : a dataframe, columns = attributeList + ['doc_id']
        """
        docs = copy.copy(textList)

        """
        for file in textList:
            tokens = settings.enc.encode(file)
            truncated_tokens = tokens[:4000]
            truncated_text = settings.enc.decode(truncated_tokens)
            docs.append(truncated_text)
        """

        attributes = ", ".join(attributeList)
        related_attr_descriptions = []
        for attr in attributeList:
            related_attr_descriptions.append(f"{attr}: {self.attr_descriptions_dict.get(attr)}")
        related_attr_descriptions_str = " \n".join(related_attr_descriptions)
        prompts = [
            [
                {"role": "system", "content": "You are an information extraction assistant. Only respond with key-value pairs using the exact field names provided. Do not include any explanations or extra text."},
                {"role": "user", "content": f'''Extract the following fields from the given document: {attributes}.

                Instructions:
                - Format your response as lines, each in the format: `field: value`
                - Use the exact field names: {attributes}
                - Follow the descriptions of the fields:
                ``` {related_attr_descriptions_str} ```
                - If a field is missing or unknown, leave its value empty (e.g., `team: None`)
                - use the line break (`\\n`)to split the lines
                - Do not add any extra text, comments, or explanations

                Document:
                {doc}
                '''
                }
            ]
            for doc in docs
        ]
        results = self.batch_llm_response(prompts=prompts) # totest

        json_result = [parse_result(results[i], doc_idList[i], attributeList) for i in range(len(results))]
        df = pd.DataFrame(json_result)
        df = df.fillna(" ")
        attributeList.append('doc_id')
        df = table_util.check_missing_columns(df, attributeList)
        #print("use prompt:", related_attr_descriptions_str)
        print("------------\n", df)
        return df
    

    def check_filter_condition(self, docs, doc_idList, attributeList, filterList):
        """
        textList : list[str], each element is a text document, corresponding to doc_idList
        doc_idList : list[str], id of the text documents, corresponding to textList
        filterList : list[str], filter conditions to apply on the extracted attributes

        extract the attributes from the textList, which is a list of text documents.

        output : a dataframe, columns = attributeList + ['doc_id'] + ['fcondition']
        """

        filters = ", ".join(filterList)
        
        related_attr_descriptions = []
        for attr in attributeList:
            related_attr_descriptions.append(f"{attr}: {self.attr_descriptions_dict.get(attr)}")
        related_attr_descriptions_str = " \n".join(related_attr_descriptions)
        prompts = [
            [
                {"role": "system", "content": "You are an condition check assistant. Respond in a single line, includes a pair format as `fcondition: True/False`, the boolean True or False represents whether the condition is met. Do not include any explanations or extra text."},
                {"role": "user", "content": f'''Check if the value satisfy the condition or semantically similar to the condition.

                Instructions:
                - Format your response as a single line, in the format: `fcondition: True/False`.
                - Check if the value satisfies the condition {filters}.
                - The filter condition `==` or `IN` can be considered emantically for strings. For example, \'Lakers\' and \'Los Angeles Lakers\' are equal, \'fashion\' and \'Fashion || Illustration\' are also equal.
                - The filter conditioin `<`, `>`, `>=`, `<=` can be considered as numeric comparison or a date comparison, note that the eariler date is smaller.
                - For example, the filed is \'birth date\', and value is \'2001/10/6\'. Then we do check, the condition is \'birth date<\'1999/11/6\' \', the value does not satisfy the condition, so the fcondition is False, output fcondition: False.
                - For example, the filed is \'style\', and value is \'fashion\'. Then we do check, the condition is \'style == Fashion\', the value satisfy the condition, so the fcondition is True, output fcondition: True.
                - Do not add any extra text, comments, quotes, or explanations.

                Value:
                {doc}

                Condition:
                {filters}
                '''
                }
            ]
            for doc in docs
        ]
        results = self.batch_llm_response(prompts=prompts) # totest

        attributeList.append('fcondition')
        json_result = [parse_result(results[i], doc_idList[i], attributeList) for i in range(len(results))]
        df = pd.DataFrame(json_result)
        df = df.fillna(" ")
        attributeList.append('doc_id')
        df = table_util.check_missing_columns(df, attributeList)
        # check columns

        print("use prompt:", related_attr_descriptions_str)
        print("------------\n", df)
        return df

    def single_iter_llm_response(self, prompts):
        results = []
        LLMInfo.add_query_times(len(prompts))
        for prompt in tqdm(prompts):
            response = completion(
                    model=self.llm, 
                    messages=prompt,
                    max_tokens=128,
                    stop=None,
                    temperature=0,
                    api_base=self.api_base
                )

            results.append(response.choices[0].message.content)
        
        for prompt in prompts:
            for talk in prompt:
                for v in talk.values():
                    LLMInfo.add_input_tokens(len(settings.enc.encode(v)))

        for v in results:
            LLMInfo.add_output_tokens(len(settings.enc.encode(v)))
        
        return results

    def batch_llm_response(self, prompts):
        results = []
        LLMInfo.add_query_times(len(prompts))

        batch_responses = batch_completion(
                model=self.llm, 
                messages=prompts,
                max_tokens=128,
                stop=None,
                temperature=0,
                api_base=self.api_base
            )
        
        for response in batch_responses:
            results.append(response.choices[0].message.content)

        for prompt in prompts:
            for talk in prompt:
                for v in talk.values():
                    LLMInfo.add_input_tokens(len(settings.enc.encode(v)))

        for v in results:
            try:
                LLMInfo.add_output_tokens(len(settings.enc.encode(v)))
            except:
                print_log("Token count error occurred.")
                print_log(v)
        
        return results          
