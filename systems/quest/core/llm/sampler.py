import re
import pandas as pd
from litellm import completion, batch_completion
from tqdm import tqdm
import tiktoken
import os
import logging
import quest.conf.settings as settings
import copy
import random
from quest.core.llm.llm_query import LLMInfo

import re

from quest.db.indexer.single_indexer import SingleIndexer, TextDocIndexer

def extract_attr_descriptions_from_schema(attr_schema):
    # 从attr_schema字符串中提取属性描述列表
    attr_descriptions = []
    lines = attr_schema.strip().split('\n')
    for line in lines:
        line = line.strip()
        if ':' in line:
            # 提取冒号后的属性描述
            attr_description = line.split(':')[1].strip()
            if attr_description:
                attr_descriptions.append(attr_description)
    return attr_descriptions



def extract_attr_names_from_schema(attr_schema):
    """
    从attr_schema字符串中提取属性名列表
    
    Args:
        attr_schema: 格式如 "name: description\nage: description\n..."
        
    Returns:
        list: 属性名列表，如 ['name', 'age', 'team', 'nba_draft_pick']
    """
    attr_names = []
    lines = attr_schema.strip().split('\n')
    for line in lines:
        line = line.strip()
        if ':' in line:
            # 提取冒号前的属性名
            attr_name = line.split(':')[0].strip()
            if attr_name:
                attr_names.append(attr_name.lower())
    return attr_names

def parse_xyz(input_str):
    #  (key, value, confidence)
    # 去除首尾空格和括号
    stripped = input_str.strip().strip('()')
    # 定义正则表达式
    pattern = r'^\s*([^,]+)\s*,\s*(.*?)\s*,\s*(\d+)\s*$'
    match = re.match(pattern, stripped)
    
    if not match:
        # print("未匹配成功的输入字符串是:\n", input_str)
        return None
    
    x = match.group(1).strip()
    y = match.group(2).strip()
    z = int(match.group(3))
    return (x, y, z)

def parse_xyz_with_chunkid(input_str, attr_names=None):
    """
    解析格式为 (key, value, confidence, chunkid) 的字符串
    
    Args:
        input_str: 输入字符串，例如 "(name, John Doe, 95, 123)"
        attr_names: 有效的属性名列表，用于验证key是否有效
        
    Returns:
        tuple: (key, value, confidence, chunkid) 或 None（如果解析失败）
    """
    # 去除首尾空格
    stripped = input_str.strip()
    
    # 严格验证括号格式：必须以'('开头，以')'结尾
    if not (stripped.startswith('(') and stripped.endswith(')')):
        print("格式错误：缺少括号或括号位置不正确。输入字符串是:\n", input_str)
        return None
    
    # 去除括号
    content = stripped[1:-1].strip()
    
    # 定义更宽松的正则表达式，允许confidence和chunkid包含非数字字符
    # 更新：这里不再限制confidence和chunkid必须是纯数字，而是接受任何非逗号字符
    pattern = r'^\s*([^,]+)\s*,\s*(.*?)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*$'
    match = re.match(pattern, content)
    
    if not match:
        print("未匹配成功的输入字符串是:\n", input_str)
        return None
    
    x = match.group(1).strip()
    y = match.group(2).strip()
    
    # 从confidence和chunkid中提取连续的数字部分
    # 更新：即使confidence和chunkid包含非数字字符，也会提取其中的数字部分
    z_str = match.group(3).strip()
    chunkid_str = match.group(4).strip()
    
    # 提取数字
    z_match = re.search(r'\d+', z_str)
    chunkid_match = re.search(r'\d+', chunkid_str)
    
    if not z_match or not chunkid_match:
        print("未能提取到有效的数字。输入字符串是:\n", input_str)
        return None
    
    z = int(z_match.group())
    chunkid = int(chunkid_match.group())
    
    # 后处理：去除key和value外面的引号
    x = x.strip('\'"')  # 去除单引号或双引号
    y = y.strip('\'"')  # 去除单引号或双引号
    
    # 验证属性名是否在允许的列表中
    if attr_names is not None:
        if x.lower() not in [name.lower() for name in attr_names]:
            print(f"属性名验证失败：'{x}' 不在允许的属性列表 {attr_names} 中。输入字符串是:\n", input_str)
            return None
    
    # 后处理：清理value中的额外信息
    if x.lower() == 'name':
        # 对于name属性，去除括号中的额外信息（如生日）
        # 匹配模式：人名 (额外信息)
        name_pattern = r'^([^(]+?)(?:\s*\([^)]+\))?$'
        name_match = re.match(name_pattern, y)
        if name_match:
            y = name_match.group(1).strip()
    
    return (x, y, z, chunkid)


class AttrSampler:

    def __init__(self, schema = "", llm = settings.GPT_MODEL, api_base= settings.GPT_API_BASE, max_tokens = 1024):
        self.api_base = settings.GPT_API_BASE # api_base
        self.llm =  settings.GPT_MODEL # llm        
        self.extract_task_prompt = """
            Your Task is to extract key-value pairs from text chunks with following guides:
            1. InPut: 
                • Schema: Attributes to be extracted and their corresponding descriptions
                • Chunks: A list of text chunks to be extracted, each marked with its ID at the beginning.
            2. Output:
                • `key`: lowercase attribute_name from schema (e.g.,name)  
                • `value`: attribute_value with exact casing/spacing (e.g., iPhone 14)
                • `confidence`: int, between 0 to 100.
                • `chunkid`: int, id of the chunk from which the key-value pair is extracted.
                • Output one tuple per line, formatted as  (attr_name, attr_value, confidence, chunkid).
            """

        self.system_prompt = "You are an attribute extraction assistant. Only respond with (key, value, confidence, chunkid) pairs. Do not include any explanations or extra text."
        self.sample_table = pd.DataFrame()
        self.map_attr_evidence = {}
        self.max_tokens = max_tokens
        self.schema = schema
        return

    def insert_table(self, doc_id, t):
        #  (key, value, confidence, evidence)
        # print(t)
        key, value, confidence, evidence_text = t
        key_confidence_str = key + "_confidence"
        key_evidence_text_str = key + "_evidence"      
        
        ################
        # 先检查key是否存在于table中，如果不存在，加上对应的key列，以及对应的confidence, evidence_text。
        # if self.sample_table.get(key) is None:
        #     new_columns = {key: [None], key_confidence_str: [None], key_evidence_text_str: [None]}  # 或其他实际值
        #     self.sample_table = pd.concat([self.sample_table, pd.DataFrame(new_columns)], axis=1)

        # 先检查key是否存在于table中，如果不存在，加上对应的key列，以及对应的confidence, evidence_text。
        for col in [key, key_confidence_str, key_evidence_text_str]:
            if col not in self.sample_table.columns:
                self.sample_table[col] = None



        ###########
        # 检查doc_id对应的行是否存在，如果不存在，加上一行
        if doc_id not in self.sample_table.index:
            self.sample_table.loc[doc_id] = [None] * len(self.sample_table.columns) # 创建这个新的行

        ##########
        # [doc_id, key] 检查value是否相同，如果不同，根据confidence替换value，更新confidence和chunksid
        pre_value = self.sample_table.loc[doc_id, key]
        if pre_value is None:
            self.sample_table.loc[doc_id, key] = value
            self.sample_table.loc[doc_id, key_confidence_str] = confidence
            self.sample_table.loc[doc_id, key_evidence_text_str] = evidence_text
        elif pre_value != value:
            if confidence > self.sample_table.loc[doc_id, key_confidence_str]:
                self.sample_table.loc[doc_id, key] = value
                self.sample_table.loc[doc_id, key_confidence_str] = confidence
                self.sample_table.loc[doc_id, key_evidence_text_str] = evidence_text
        return

        #############
        # 如果value相同，根据confidence替换confidence，更新chunksid


    def response_single_doc(self, chunks, chunks_id, attr_Schema):
        
        chunks_to_extract = ""
        for i, text in enumerate(chunks):
            chunks_to_extract += f'''
            Chunk_id {chunks_id[i]}:  
            ```  
            {text}
            ```  

            '''

        attr_prompt = f'''
            Schema:  
            ```
            {attr_Schema}
            ```         
            '''
        
        text_prompt = f'''
            Chunks:  
            ```  
            {chunks_to_extract}
            ```  
            '''
        
        
        user_prompt =  self.extract_task_prompt +  attr_prompt + text_prompt
        final_prompt = [
            {"role": "system", "content": self.system_prompt}, 
            {"role": "user", "content": user_prompt}
            ]        
        
        LLMInfo.add_query_times(1)
        for talk in final_prompt:
                for v in talk.values():
                    LLMInfo.add_input_tokens(len(settings.enc.encode(v)))

        response = completion(
                model=self.llm, 
                messages=final_prompt,
                max_tokens=self.max_tokens,
                stop=None,
                temperature=0,
                api_base=self.api_base
            )

        result = response.choices[0].message['content'].strip()     
        LLMInfo.add_output_tokens(len(settings.enc.encode(result)))

        return result        

    def extract_doc2row(self, doc_id, chunks, attr_Schema):
        """
        - doc_id: 当前抽取的被采样文档的编号
        - doc_chunks_list: 当前被采样文档的分块列表
        - chunks_id: 当前被采样文档的分块在文档全集分块列表中的编号        
        """
        # 从schema中提取属性名列表
        attr_names = extract_attr_names_from_schema(attr_Schema)
        chunks_id = list(range(len(chunks)))

        result = self.response_single_doc(chunks, chunks_id, attr_Schema= attr_Schema)
        tuples = result.split("\n")

        for t in tuples:
            t = parse_xyz_with_chunkid(t, attr_names=attr_names)
            if t is None:
                continue
            if t[1] is None:
                continue
            if t[2] < 50:
                continue
            if t[3] >= len(chunks) or t[3] < 0:
                continue
            evidence_text = chunks[t[3]]
            new_tuple = (t[0], t[1], t[2], evidence_text)
        # (name, Donald Trump, 100, 91)
        # (key, value, confidence, chunksid)                
            # 以doc_id为主键
            self.insert_table(doc_id, new_tuple)

    def sample_one_doc(self, doc_id, doc_indexer : TextDocIndexer, attr_schema):
        chunks = doc_indexer.get_chunks_by_docid(doc_id)
        self.extract_doc2row(doc_id, chunks, attr_schema)
        return

    def try_sample(self, doc_indexer: TextDocIndexer, attr_schema):
        """
        根据schema从indexer对应的文档中采样获得sample表
        """
        doc_ids = doc_indexer.get_docs_id()
        N = len(doc_ids)
        sample_num = min(N, max(settings.SAMPLE_NUM, int(N/20)))
        sampler_ids = random.sample(doc_ids, sample_num)
        for doc_id in sampler_ids:
            self.sample_one_doc(doc_id, doc_indexer, attr_schema)
        self.map_attr_evidence = self.get_evidence(attr_schema)

    def get_evidence(self, attr_schema = ""):
        if len(attr_schema)<10:
            attr_schema = copy.copy(self.schema)
        map_attr_evidence = {} #
        attr_names = extract_attr_names_from_schema(attr_schema)
        for attr in attr_names:
            value_col = attr
            confidence_col = f"{attr}_confidence"
            evidence_col = f"{attr}_evidence"
            if (
                value_col not in self.sample_table.columns or 
                confidence_col not in self.sample_table.columns or
                evidence_col not in self.sample_table.columns
            ):
                map_attr_evidence[attr] = ""
                continue
            
            # 取信心度不为None的行
            valid_rows = self.sample_table[self.sample_table[confidence_col].notnull()]
            if len(valid_rows) == 0:
                map_attr_evidence[attr] = ""
                continue
            # 根据confidence排序，取top2
            top2 = valid_rows.sort_values(by=confidence_col, ascending=False).head(2)
            # 拼接evidence文本（有可能重复/为None，要做去重和非空处理）
            evidences = [str(e) for e in top2[evidence_col].tolist() if e and str(e).strip()]

            concated_ev = "---------------------------------\n".join(evidences)
            map_attr_evidence[attr] = concated_ev
        return map_attr_evidence

    def get_attr_schema_evidence(self, attr_schema = ""):
        if len(attr_schema)<10:
            attr_schema = copy.copy(self.schema)        
        map_attr_evidence = {}

        attr_names = extract_attr_names_from_schema(attr_schema)
        attr_descriptions = extract_attr_descriptions_from_schema(attr_schema)
        for  attr, description in zip(attr_names, attr_descriptions):
            map_attr_evidence[attr] = attr + " : " + description
        return  map_attr_evidence

