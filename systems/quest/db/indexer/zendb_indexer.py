#!/usr/bin/env python3
"""
ZenDB 完整实现 - 基于论文设计的学术文档分析系统
"""

MAX_SENTENCES = 10

import pdfplumber
import sqlite3
import openai
import re
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime
import time
from openai import OpenAI
import random
from quest.core.datapack.doc import ZenDBDoc
from quest.core.llm.llm_query import LLMInfo
from quest.core.nlp.doc_summary import tfidf_summary
from quest.db.indexer.single_indexer import SingleIndexer
from quest.db.querier.querier import OpenGaussQuerier
import wordninja
import pyparsing as pp
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import faiss

# 设置DeepSeek API密钥和配置
DEEPSEEK_API_KEY = 'sk-DrStrjrlXohoCI2iBc14Eb41A38c428b88A0973aA29fCc3b'
DEEPSEEK_BASE_URL = 'https://aihubmix.com/v1'

# 初始化DeepSeek客户端
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)

@dataclass
class SHTNode:
    """语义层次树节点"""
    node_id: int
    doc_id: int
    name: str
    granularity: int
    context: str
    summary: str
    start_pos: int
    end_pos: int
    parent: Optional['SHTNode'] = None
    children: List['SHTNode'] = None
    visual_pattern: Dict[str, Any] = None
    full_context: str = None  # 新增字段，包含本节点及所有子节点的全文
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.full_context is None:
            self.full_context = self.context
    
    def to_dict(self):
        return {
            "node_id": self.node_id,
            "doc_id": self.doc_id,
            "name": self.name,
            "granularity": self.granularity,
            "context": self.context,
            "summary": self.summary,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "visual_pattern": self.visual_pattern,
            # "full_context": self.full_context, # 不需要存full_text，太大
            "children": [child.to_dict() for child in self.children],
        }

    @staticmethod
    def from_dict(data, parent=None):
        node = SHTNode(
            node_id= int(data["node_id"]),
            doc_id=int(data["doc_id"]),
            name=data["name"],
            granularity=data["granularity"],
            context=data["context"],
            summary=data.get("summary", ""),
            start_pos= int(data.get("start_pos", 0)),
            end_pos= int(data.get("end_pos", 0)) ,
            parent=parent,
            children=[],
            visual_pattern=data.get("visual_pattern"),
            # full_context=data.get("full_context")
        )
        node.children = [SHTNode.from_dict(child, parent=node) for child in data.get("children", [])]
        return node


    def get_ancestors(self):
        """获取祖先节点路径"""
        ancestors = []
        current = self.parent
        while current:
            ancestors.append(current.name)
            current = current.parent
        return ancestors[::-1]
    
    def is_leaf(self):
        """判断是否为叶子节点"""
        return len(self.children) == 0

from quest.conf.settings import ABS_PROJECT_ROOT_PATH, count_tokens

class ZenDBDocIndexer(SingleIndexer):
    """ZenDBIndexer主类"""
    
    def __init__(self, table_name: str, type: str = "ZenDBDoc", root_save_path = os.path.join(ABS_PROJECT_ROOT_PATH, "data/single_index"), need_clean_chunk = False, embedding_model = None, **kwargs):
        print("🚀 建立ZenDB-singleIndex索引...")
        self.embedding_size = embedding_model.emb_size 
        self.embedding_model = embedding_model
        self.need_clean_chunk = need_clean_chunk
        self.table_name = table_name
        self.type = type

        self.table_save_path = os.path.join(root_save_path, f"{self.table_name}.json")
        self.embedding_save_path = os.path.join(root_save_path, f"{self.table_name}_embeddings.pkl")
        self.sht_tables = {}  # doc_id: int -> SHT nodes
        self.user_tables = {}  # table_name -> SQLite table
        self.templates = []    # 跨文档模板缓存
        # 新增embedding相关成员变量
        self.node_embeddings: Dict[int, np.ndarray] = {}  # node_id -> embedding向量
        # self.embedding_cache: Dict[str, np.ndarray] = {}  # 缓存相同文本的embedding
        self.query_embedding_cache: Dict[str, np.ndarray] = {}  # 缓存相同文本的embedding
        # self.querier = OpenGaussQuerier()        

    def build_indexer(self, docs: List[ZenDBDoc]) -> None:
        # 提取并存储docs_meta
        self.docs_meta = {doc["doc_id"]: doc.metadata for doc in docs}
        # self.querier.build_cache_table(self.table_name, self.docs_meta)

        self.build_sht_for_all_docs(docs)
        # 为所有节点生成embedding
        self._generate_embeddings_for_all_docs()
        self.save_indexer()
        return

    def save_indexer(self):
        # 保存SHT树结构
        path = self.table_save_path
        sht_dict = {doc_id: root.to_dict() for doc_id, root in self.sht_tables.items()}
        data = {
            "docs_meta": self.docs_meta,
            "sht_trees": sht_dict
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # 保存embedding数据
        with open(self.embedding_save_path, "wb") as f:
            pickle.dump(self.node_embeddings, f)

    def load_indexer(self):
        # 加载SHT树结构
        path = self.table_save_path
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.docs_meta = data["docs_meta"]
        # 把字典中所有key转换成int类型
        self.docs_meta = {int(k): v for k, v in self.docs_meta.items()}
        self.sht_tables = {
            int(doc_id) : SHTNode.from_dict(root_dict)
            for doc_id, root_dict in data["sht_trees"].items()
        }
        
        # 加载embedding数据
        if os.path.exists(self.embedding_save_path):
            with open(self.embedding_save_path, "rb") as f:
                self.node_embeddings = pickle.load(f)
        return

    def _generate_embeddings_for_all_docs(self):
        """为所有文档的所有节点生成embedding"""
        print("\n🧠 生成节点embedding...")
        total_nodes = 0
        for doc_id, root in self.sht_tables.items():
            self._generate_embeddings(root)
            total_nodes += self._count_nodes(root)
        print(f"  ✓ 完成 {total_nodes} 个节点的embedding生成")

    def _generate_embeddings(self, node: SHTNode) -> None:
        """为节点生成embedding - 使用批量嵌入提高效率"""
        # 获取所有节点的context和node_id列表
        context_list, node_id_list = self.level_traverse(node.doc_id)
        
        if not context_list:
            return
        
        # 批量生成embedding
        embeddings = self.embedding_model.embed_documents(context_list)
        
        # 构建node_id到embedding的映射
        for node_id, embedding in zip(node_id_list, embeddings):
            self.node_embeddings[node_id] = embedding

    def _get_embedding(self, node_id: int) -> np.ndarray:
        """获取节点的context的embedding向量"""
        return self.node_embeddings.get(node_id, None)

    def _get_cached_query_embedding(self, query: str):
        """
        获取缓存的query embedding，如果不存在则计算并缓存
        
        Args:
            query: 查询字符串
            
        Returns:
            query的embedding向量
        """
        cache_key = query[:256]  # 使用前256个字符作为缓存键
        
        if cache_key not in self.query_embedding_cache:
            # 计算embedding并缓存
            self.query_embedding_cache[cache_key] = self.embedding_model.embed_query(query)
        
        return self.query_embedding_cache[cache_key]


    def _semantic_similarity_search(self, query: str, node_ids: List[int], topk: int) -> List[int]:
        """在指定节点集合中进行语义相似度搜索（使用faiss加速）"""
        if not node_ids:
            return []
        
        query_embedding = self._get_cached_query_embedding(query)
        # query_embedding = self.embedding_model.embed_query(query)
        # 收集节点embedding
        embeddings = []
        valid_node_ids = []
        for node_id in node_ids:
            emb = self._get_embedding(node_id)
            if emb is not None:
                embeddings.append(emb)
                valid_node_ids.append(node_id)
            else:
                raise ValueError(f"节点 {node_id} 没有embedding")
        if not embeddings:
            return []
        # 转为np数组
        xb = np.array(embeddings).astype('float32')
        xq = np.array([query_embedding]).astype('float32')
        # 归一化（余弦相似度）
        faiss.normalize_L2(xb)
        faiss.normalize_L2(xq)
        # 建立faiss索引
        index = faiss.IndexFlatIP(xb.shape[1])
        index.add(xb)
        D, I = index.search(xq, min(topk, xb.shape[0]))
        # 返回对应的node_id
        result_ids = [valid_node_ids[i] for i in I[0]]
        return result_ids

    def build_indexer(self, docs: List[ZenDBDoc]) -> None:
        # 提取并存储docs_meta
        self.docs_meta = {doc["doc_id"]: doc.metadata for doc in docs}
        # self.querier.build_cache_table(self.table_name, self.docs_meta)

        self.build_sht_for_all_docs(docs)
        # 为所有节点生成embedding
        self._generate_embeddings_for_all_docs()
        self.save_indexer()
        return

    def save_indexer(self):
        # 保存SHT树结构
        path = self.table_save_path
        sht_dict = {doc_id: root.to_dict() for doc_id, root in self.sht_tables.items()}
        data = {
            "docs_meta": self.docs_meta,
            "sht_trees": sht_dict
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # 保存embedding数据
        with open(self.embedding_save_path, "wb") as f:
            pickle.dump(self.node_embeddings, f)

    def load_indexer(self):
        # 加载SHT树结构
        path = self.table_save_path
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.docs_meta = data["docs_meta"]
        # 把字典中所有key转换成int类型
        self.docs_meta = {int(k): v for k, v in self.docs_meta.items()}
        self.sht_tables = {
            int(doc_id) : SHTNode.from_dict(root_dict)
            for doc_id, root_dict in data["sht_trees"].items()
        }
        
        # 加载embedding数据
        if os.path.exists(self.embedding_save_path):
            with open(self.embedding_save_path, "rb") as f:
                self.node_embeddings = pickle.load(f)
        return

    def get_file_name_by_id(self, doc_id: int) -> str:
        """
        根据文档ID获取对应的文件名

        Args:
            doc_id: 文档编号

        Returns:
            str: 对应的文件名
        """
        if doc_id in self.docs_meta:
            return self.docs_meta[doc_id].get("file_name", "")
        return ""

    def get_chunks_by_docid(self, doc_id):
        chunk_list, id_list = self.level_traverse(doc_id)
        return chunk_list

    def level_traverse(self, doc_id):
        """
        层序遍历指定文档的分块树，返回 (context_list, node_id_list)。
        """
        root = self.sht_tables.get(doc_id)
        if root is None:
            return [], []
        context_list = []
        node_id_list = []
        queue = [] 
        queue.extend(root.children) # 不要把root节点的context返回到结果中，因为它包含全文。
        while queue:
            node = queue.pop(0)
            context_list.append(node.context)
            node_id_list.append(node.node_id)
            queue.extend(node.children)
        return context_list, node_id_list


    def print_sht_tree(self, doc_id):
        """
        以前序遍历的形式递归地打印整个SHT分块树的结构，按节点所在的深度进行缩进。
        
        Args:
            doc_id: 文档编号
        """
        root = self.sht_tables.get(doc_id)
        if root is None:
            print(f"文档 {doc_id} 的SHT树不存在")
            return
        
        def _print_node(node, depth=0):
            """递归打印节点及其子节点"""
            indent = "  " * depth
            print(f"{indent}node_id: {node.node_id}")
            for child in node.children:
                _print_node(child, depth + 1)
        
        print(f"文档 {doc_id} 的SHT树结构:")
        _print_node(root)


    def llm_judge_contain_attr(self, node, query):
        """
        调用LLM，根据当前节点的summary判断当前节点是否可能包含query中需要提取的属性，
        llm只给出true或false的结果以及结果的confidence(0到100的整数)，
        返回 (bool, int)，如果格式不对则默认(False, 100)，并统计token消耗。
        """
        try:
            # 构造提示
            prompt = (
                f"Determine if the following section contains the attribute defined in the query based on the summary of the section.\n"
                f"Query: {query}\n"
                f"Section Title: {node.name}\n"
                f"Section Summary: {node.summary}\n"
                f"Answer format: true or false, confidence from 0 to 100. e.g. (true, 100)"
            )
            # 统计输入输出token（简易估算）
            input_tokens = count_tokens(prompt)
            response = self._llm_query(prompt)
            output_tokens = count_tokens(response)
            # 更新token计数和调用次数计数
            LLMInfo.add_input_tokens(input_tokens)
            LLMInfo.add_output_tokens(output_tokens)
            LLMInfo.add_query_times(1)

            # 解析返回结果
            m = re.search(r"(true|false)\s*[,，]\s*(\d{1,3})", response.lower(), re.IGNORECASE)
            if m:
                result = m.group(1).lower() == 'true'
                confidence = min(max(int(m.group(2)), 0), 100)
                return result, confidence
        except Exception:
            pass
        # 默认值
        return False, 0

    def pre_order_dfs_search_related_node(self, node, query):
        """
        基于DFS的前序遍历，判断并收集可能包含查询属性的节点：
          - 若 node 为空或 llm_judge_contain_attr 返回 False，返回 False
          - 若 llm_judge_contain_attr 返回 True，则对子节点递归调用，
            当且仅当所有子节点递归均返回 False 时，将当前节点加入候选
        返回是否该 subtree 可能包含属性的 bool
        高级剪枝策略：利用topK，如果当前判断包含属性的块数已经达到topK，则直接返回。
        """
        if node is None:
            return False
        contain, confidence = self.llm_judge_contain_attr(node, query)
        if not contain:
            return False
        # 标记子节点中是否有命中
        child_hit = False
        for child in node.children:
            if self.pre_order_dfs_search_related_node(child, query):
                child_hit = True
        # 若子节点都未命中，则将当前节点作为候选
        if not child_hit:
            if not hasattr(self, 'candidate_nodes_with_confidence'):
                self.candidate_nodes = []
            self.candidate_nodes.append((node, confidence))
        return True


    def topk_semantic_beam_search_related_node(self, node: SHTNode, query: str, beam_topk: int) -> bool:
        """基于层次化束搜索的节点搜索方法"""
        self.candidate_nodes = []
        only_one_1_level_title_flag = False
        if beam_topk <= 0:
            return False
        
        
        # 第一阶段：收集所有granularity=1的节点
        # granularity_1_nodes = []
        # self._collect_nodes_by_granularity(node, 1, granularity_1_nodes)
        # 直接取node(即root节点)的所有子节点作为granularity=1的节点
        # jztodo 如果1级标题只有1个，则把2级标题作为granularity_1_nodes。
        granularity_1_nodes = node.children
        if len(granularity_1_nodes) == 1:
            granularity_1_nodes = granularity_1_nodes[0].children
            only_one_1_level_title_flag = True
        
        if len(granularity_1_nodes) == 0 or not granularity_1_nodes:
            # 如果没有granularity=1的节点，直接认为整篇文章不包含attr，返回False
            return False
        
        # 按语义相似度排序granularity=1的节点，取topK个进行LLM判断
        node_ids = [n.node_id for n in granularity_1_nodes]
        sorted_node_ids = self._semantic_similarity_search(query, node_ids, beam_topk)
        
        # 第二阶段：LLM判断语义相似度最高的topK个granularity=1节点
        V1 = []  # 存储被LLM判定为true的granularity=1节点
        for node_id in sorted_node_ids:
            # 找到对应的节点
            target_node = None
            for n in granularity_1_nodes:
                if n.node_id == node_id:
                    target_node = n
                    break
            
            if target_node:
                contain, confidence = self.llm_judge_contain_attr(target_node, query)
                if contain:
                    V1.append(target_node)
        
        # 第三、四阶段：层次化收集子节点
        all_candidate_nodes = V1.copy()  # 从V1开始
        current_level_nodes = V1
        
        while current_level_nodes:
            # 收集当前层级所有节点的子节点
            next_level_nodes = []
            for parent_node in current_level_nodes:
                next_level_nodes.extend(parent_node.children)
            
            if not next_level_nodes:
                break
            
            # 按语义相似度排序，取topK个
            child_node_ids = [n.node_id for n in next_level_nodes]
            sorted_child_ids = self._semantic_similarity_search(query, child_node_ids, beam_topk)
            
            # 获取topK个子节点
            topk_children = []
            for node_id in sorted_child_ids:
                for child in next_level_nodes:
                    if child.node_id == node_id:
                        topk_children.append(child)
                        break
            
            # 添加到候选列表
            all_candidate_nodes.extend(topk_children)
            current_level_nodes = topk_children
        
        # 第五阶段：合并所有候选节点，按语义相似度排序，取最终topK
        if all_candidate_nodes:
            # 去重（避免重复节点）
            unique_nodes = []
            seen_ids = set()
            for node in all_candidate_nodes:
                if node.node_id not in seen_ids:
                    unique_nodes.append(node)
                    seen_ids.add(node.node_id)
            
            # 按语义相似度排序 jztodo 如果1级标题就1个，把它也加入到列表中
            final_node_ids = [n.node_id for n in unique_nodes]
            if only_one_1_level_title_flag:
                final_node_ids.append(granularity_1_nodes[0].node_id)
            final_sorted_ids = self._semantic_similarity_search(query, final_node_ids, beam_topk)
            
            # 取topK个节点，进行最终的LLM判断并添加到候选列表
            for i, node_id in enumerate(final_sorted_ids[:beam_topk]):
                # 找到对应的节点
                target_node = None
                for n in unique_nodes:
                    if n.node_id == node_id:
                        target_node = n
                        unique_nodes.remove(n)
                        break
                
                if target_node:
                    self.candidate_nodes.append(target_node)
        
        return len(self.candidate_nodes) > 0

    def _collect_nodes_by_granularity(self, node: SHTNode, target_granularity: int, result_list: List[SHTNode]):
        """递归收集指定granularity的节点"""
        if node.granularity == target_granularity:
            result_list.append(node)
        
        # 继续递归搜索子节点
        for child in node.children:
            self._collect_nodes_by_granularity(child, target_granularity, result_list)


    def topk_query_all_nodes(self, doc_id: int, query: str, topk: int) -> List[tuple[str, int]]:
        """
        对指定文档的所有节点进行语义相似度搜索，返回最相关的topK个节点
        
        Args:
            doc_id: 文档ID
            query: 查询文本
            topk: 返回的节点数量
            
        Returns:
            List[tuple[str, int]]: [(node_context, node_id), ...] 按相似度降序排列
        """
        # 获取根节点
        root = self.sht_tables.get(doc_id)
        if root is None:
            return []
        
        # 使用level_traverse获取所有节点
        context_list, node_id_list = self.level_traverse(doc_id)
        
        if not node_id_list:
            return []
        
        # 对所有节点进行语义相似度搜索
        sorted_node_ids = self._semantic_similarity_search(query, node_id_list, topk)
        
        # 构造结果列表
        results = []
        node_id_to_context = dict(zip(node_id_list, context_list))
        
        for node_id in sorted_node_ids:
            context = node_id_to_context.get(node_id, "")
            results.append((context, node_id))
        
        return results




    def get_relative_chunks_text_with_id(self, doc_id: int, query: str, topk: int) -> List[tuple[str, int]]:
        """
        查询指定文档中与query相关的topk个文本块
        返回 [(chunk_text, chunk_id), ...]，chunk_id为块起始位置
        """
        # 获取根节点
        root = self.sht_tables.get(doc_id)
        if root is None:
            return []
        
        # 重置候选列表
        self.candidate_nodes = []
        
        # 只有根节点的一级子节点会参与判断，其他节点都是直接用束搜索取topk。
        has_candidate_flag = self.topk_semantic_beam_search_related_node(root, query, topk)  
        # 如果has_candidate_flag为False，退化成对所有树节点进行topK语义相似度搜索。先使用level_traversal获得所有节点，然后进行topK语义相似度搜索。        
        if not has_candidate_flag:
            return self.topk_query_all_nodes(doc_id, query, topk)    

        topk_nodes = self.candidate_nodes 
        
        # 构造结果列表
        results = []
        for node  in topk_nodes:
            results.append((node.context, node.node_id))
        
        
        return results

    def depreacated_topk_semantic_dfs_search_related_node(self, node: SHTNode, query: str, remaining_topk: int) -> bool:
        """新的基于语义相似度的DFS搜索方法"""
        if remaining_topk <= 0:
            return False
        
        # 对当前节点进行LLM判断
        contain, confidence = self.llm_judge_contain_attr(node, query)
        
        if contain and confidence >= 70:
            # 如果当前节点包含属性，加入候选集合
            if not hasattr(self, 'candidate_nodes_with_confidence'):
                self.candidate_nodes = []
            
            if len(self.candidate_nodes) < remaining_topk:
                self.candidate_nodes.append((node, confidence))
                remaining_topk -= 1
        
        # 如果还需要更多候选节点且有子节点，继续搜索
        if remaining_topk > 0 and node.children:
            # 使用语义相似度对子节点排序
            child_ids = [child.node_id for child in node.children]
            sorted_child_ids = self._semantic_similarity_search(query, child_ids, len(child_ids))
            
            # 按相似度顺序处理子节点
            for child_id in sorted_child_ids:
                if remaining_topk <= 0:
                    break
                
                # 找到对应的子节点
                child_node = None
                for child in node.children:
                    if child.node_id == child_id:
                        child_node = child
                        break
                
                if child_node:
                    if self.depreacated_topk_semantic_dfs_search_related_node(child_node, query, remaining_topk):
                        remaining_topk = max(0, remaining_topk - len(self.candidate_nodes))
        
        return contain

    def _clean_text(self, text):
            """深度清洗文本内容，去除PDF转换乱码和无效字符"""
            if not text:
                return ""
            
            # 1. 移除PDF字符编码乱码 (cid:XX) 格式
            text = re.sub(r'\(cid:\d+\)', '', text)
            
            # 2. 移除其他常见PDF乱码模式
            text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)  # 控制字符
            text = re.sub(r'�', '', text)  # Unicode替换字符
            
            # 3. 移除重复的空白字符
            text = re.sub(r'\s+', ' ', text)  # 多个空格合并为一个
            text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # 多个连续换行最多保留两个
            
            # 4. 清理行首行尾空格
            lines = text.split('\n')
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if line:  # 跳过空行
                    cleaned_lines.append(line)
            
            text = '\n'.join(cleaned_lines)
            
            # 5. 移除明显的页眉页脚模式（单独的数字可能是页码）
            lines = text.split('\n')
            filtered_lines = []
            for line in lines:
                # 跳过可能的页码（单独的数字或短横线）
                if re.match(r'^\s*[-\d\s]+\s*$', line) and len(line.strip()) < 5:
                    continue
                # 跳过只有特殊字符的行
                if re.match(r'^[\s\-_=~`!@#$%^&*(){}[\]|\\:";\'<>?,.]+$', line):
                    continue
                filtered_lines.append(line)
            
            text = '\n'.join(filtered_lines)
            
            # 6. 最终格式化
            text = text.strip()
            
            # 确保文本有意义（至少有一些字母数字字符）
            if len(re.findall(r'[a-zA-Z0-9]', text)) < 10:
                return ""
            
            return text


    def _extract_phrases(self, doc):
        def smart_split_phrase(phrase):
        # 匹配编号+正文
            m = re.match(r'^((?:\d+\.)*\d+)\s*([A-Za-z]+.*)', phrase)
            if m:
                prefix = m.group(1)
                rest = m.group(2)
                # 对正文部分分词
                rest_split = ' '.join(wordninja.split(rest))
                return f"{prefix} {rest_split}"
            # 只对全英文长串分词
            if re.match(r'^[A-Za-z]{10,}$', phrase):
                return ' '.join(wordninja.split(phrase))
            return phrase
            
        """根据视觉特征连续合并短语，符合论文定义"""
        words = doc["words"]
        if (not words) or  (doc["file_type"] == "txt") or (doc["file_type"] == "md"):
                # 针对txt和md，按md标题来分phrases
            line  = doc["text"]
            phrases = []
            lines = [line.strip() for line in doc["text"].splitlines() if line.strip()]

            normal_features = {
                "size": 12,  # 默认
                "fontname": "normal_text",
                "type": {"bold": False, "underline": False},
                "all_cap": line.isupper(),
                "num_st": line[:1].isdigit(),
                "alpha_st": line[:1].isalpha(),
                "center": False
            }

            title_features = {
                "size": 12,  # 默认
                "fontname": "title",
                "type": {"bold": True, "underline": False},
                "all_cap": line.isupper(),
                "num_st": line[:1].isdigit(),
                "alpha_st": line[:1].isalpha(),
                "center": False
            }

            #####################################


            i = 0
            while i < len(lines):
                line = lines[i]
                # 检查是否为markdown标题
                header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
                
                if header_match:
                    # 如果发现是markdown标题
                    header_level = len(header_match.group(1))  # 获取#的数量
                    header_text = line  # 获取完整标题，包括前面的#
                    
                    # 设置title特征，fontname为title{level}
                    features = title_features.copy()
                    features["fontname"] = f"title{header_level}"
                    features["type"]["bold"] = True
                    
                    # 添加到phrases
                    phrases.append((header_text, features))
                    i += 1
                else:
                    # 不是标题，收集连续的非标题文本
                    normal_text = line
                    normal_features_copy = normal_features.copy()
                    
                    # 向前收集连续的非标题行
                    j = i + 1
                    while j < len(lines):
                        next_line = lines[j]
                        # 检查下一行是否为标题
                        if re.match(r'^(#{1,6})\s+(.+)$', next_line):
                            break  # 遇到标题行，停止合并
                        normal_text += '\n' + next_line  # 合并非标题行
                        j += 1
                    
                    # 添加合并后的正文
                    phrases.append((normal_text, normal_features_copy))
                    
                    # 跳过已处理的行
                    i = j

            return phrases
            # for line in lines:
            #     features = normal_features

            #     # jztodo 如果发现是markdown标题，先把之前的normal_lines(如果有的话)合并成一个phrase，然后再把title_features的fontname设置成title{n}，n是markdown标题的级别, type -> bold设置成True，然后append到phrases里面
            #     # 如果不是标题，则和其他的line合并。             
            #     phrases.append((line, features))

            # return phrases


            #################################

        phrases = []
        current_phrase = ""
        current_features = None
        current_visual = None

        def get_word_visual(word):
            # 只考虑字体名、大小、类型（加粗/下划线）
            fontname = word.get("fontname", "")
            size = word.get("size", 12)
            is_bold = int("bold" in fontname.lower() or "Bold" in fontname)
            is_underline = int("underline" in fontname.lower())
            return (size, fontname, is_bold, is_underline)

        for i, word in enumerate(words):
            word_text = word.get("text", "")
            visual = get_word_visual(word)

            if current_phrase and visual != current_visual:
                # 生成短语级视觉特征
                phrase_text = current_phrase.strip()
                # 自动在编号和正文之间加空格
                phrase_text = smart_split_phrase(phrase_text)
                
                features = {
                    "size": current_visual[0],
                    "fontname": current_visual[1],
                    "type": {
                        "bold": bool(current_visual[2]),
                        "underline": bool(current_visual[3])
                    },
                    "all_cap": phrase_text.isupper(),
                    "num_st": phrase_text[:1].isdigit(),
                    "alpha_st": phrase_text[:1].isalpha(),
                    "center": False  # 可根据x0/x1判断是否居中
                }
                phrases.append((phrase_text, features))
                current_phrase = word_text
                current_visual = visual
            else:
                current_phrase += ("" if current_phrase else "") + word_text
                current_visual = visual

        # 最后一个短语
        if current_phrase:
            phrase_text = current_phrase.strip()
            features = {
                "size": current_visual[0],
                "fontname": current_visual[1],
                "type": {
                    "bold": bool(current_visual[2]),
                    "underline": bool(current_visual[3])
                },
                "all_cap": phrase_text.isupper(),
                "num_st": phrase_text[:1].isdigit(),
                "alpha_st": phrase_text[:1].isalpha(),
                "center": False
            }
            phrases.append((phrase_text, features))


        return phrases
    
    def _build_sht(self, doc):
        return self._build_markdown_sht(doc)



    def _build_markdown_sht(self, doc):
        # jzmod
        """构建语义层级树"""
        print(f"\n🏗️ 为文档 '{doc['name']}' 构建SHT...")
        
        # 1. 提取短语
        phrases = self._extract_phrases(doc)
        print(f"  ✓ 提取 {len(phrases)} 个短语")
        
        # 2. 识别标题短语
        headers = self._identify_headers(phrases)
        print(f"  ✓ 识别 {len(headers)} 个标题")
        
        # 3. 构建树结构
        root = self._build_tree_structure(headers, doc)
        print(f"  ✓ 构建完成，根节点包含 {len(root.children)} 个子节点")
        
        # 4. 生成摘要
        self._generate_summaries(root)
        print(f"  ✓ 生成节点摘要")
        
        return root

    def _identify_headers(self, phrases):
        # 直接根据features判断哪些是标题。
        headers = []
        for phrase, features in phrases:
            if self._is_markdown_header(features):
                headers.append((phrase, features))

        return headers

    def _build_pdf_sht(self, doc):
        """构建语义层次树"""
        print(f"\n🏗️ 为文档 '{doc['name']}' 构建SHT...")
        
        # 1. 提取短语
        phrases = self._extract_phrases(doc)
        print(f"  ✓ 提取 {len(phrases)} 个短语")
        
        # 2. 识别标题短语
        headers = self._identify_pdf_headers(phrases)
        print(f"  ✓ 识别 {len(headers)} 个标题")
        
        # 3. 构建树结构
        root = self._build_pdf_tree_structure(headers, doc)
        print(f"  ✓ 构建完成，根节点包含 {len(root.children)} 个子节点")
        
        # 4. 生成摘要
        self._generate_summaries(root)
        print(f"  ✓ 生成节点摘要")
        
        return root
    
    def _identify_pdf_headers(self, phrases, k=10, n_clusters=8):
        """根据视觉特征聚类+LLM识别标题短语，符合论文定义"""
        if not phrases:
            return []

        # 先用启发式规则过滤明显不是标题的短语
        # phrases = [(phrase, info) for phrase, info in phrases if self._is_likely_header(phrase)]
        if not phrases:
            return []

        # 1. 提取视觉特征向量
        features = []
        for phrase, info in phrases:
            size = info['size']
            fontname = hash(info['fontname']) % 10000
            bold = int(info['type']['bold'])
            underline = int(info['type']['underline'])
            features.append([size, fontname, bold, underline])

        # 2. 聚类
        kmeans = KMeans(n_clusters=min(n_clusters, len(phrases)), random_state=42)
        cluster_ids = kmeans.fit_predict(features)

        # 3. 给每个phrase加cluster id
        for idx, (phrase, info) in enumerate(phrases):
            info['cluster'] = int(cluster_ids[idx])

        # 4. 按cluster分组
        cluster_groups = {}
        for phrase, info in phrases:
            cluster = info['cluster']
            if cluster not in cluster_groups:
                cluster_groups[cluster] = []
            cluster_groups[cluster].append((phrase, info))

        headers = []
        headers_set = set()
        for cluster, cluster_phrases in cluster_groups.items():
            # 采样前再用启发式过滤
            # filtered_cluster_phrases = [p for p in cluster_phrases if self._is_likely_header(p[0])]
            filtered_cluster_phrases = cluster_phrases
            if not filtered_cluster_phrases:
                continue
            sample = random.sample(filtered_cluster_phrases, min(k, len(filtered_cluster_phrases)))
            header_votes = 0
            for phrase, info in sample:
                prompt = (
        "You are analyzing a document. "
        "Section headers in a document typically indicate the start of a new section or subsection, such as chapter titles, main headings, or important topic divisions. "
        "Section headers are often short, descriptive, and may be formatted differently from the main text. "
        "Given the following phrase from the document:\n\n"
        f'"{phrase}"\n\n'
        "Is this phrase a section or subsection header in the document? "
        "Answer only 'yes' or 'no'."
)

                try:
                    response = self._llm_query(prompt)
                    #print(f"LLM判断: '{phrase}' -> {response}")
                    if response.strip().lower().startswith('y'):
                        header_votes += 1
                except Exception as e:
                    print(f"LLM异常: {e}")
                    continue
            if header_votes > len(sample) / 2:
                for phrase, info in filtered_cluster_phrases:
                    if phrase not in headers_set:
                        headers.append((phrase, cluster))
                        headers_set.add(phrase)
        return headers
    
    def _is_markdown_header(self, features):
        if  "title" in features["fontname"]:
            return True
        else:
            return False

    def _is_likely_header(self, phrase):
        """启发式规则判断是否为标题"""
        phrase = phrase.strip()
        
        # 标题特征

        if len(phrase) < 5 or len(phrase) > 100:
            return False
            
        # 包含常见标题模式
        header_patterns = [
            r'^\d+\.?\s+',  # 数字开头
            r'^[A-Z][A-Z\s]+$',  # 全大写
            r'(abstract|introduction|related work|methodology|experiments|conclusion)',  # 常见标题词
            r'^(figure|table|algorithm)\s+\d+',  # 图表标题
        ]
        
        for pattern in header_patterns:
            if re.search(pattern, phrase, re.IGNORECASE):
                return True
                
        return False
    
    def _deprecated2_build_pdf_tree_structure(self, headers, doc):
        """构建树形结构"""
        # 创建根节点
        root = SHTNode(
            node_id= 0 , # f"{doc['id']}_root",
            doc_id=doc['doc_id'],
            name=f"Document: {doc['name']}",
            granularity=0,
            context=doc['text'][:500],  # 前500字符作为上下文
            summary="",
            start_pos=0,
            end_pos=len(doc['text'])
        )
        
        if not headers:
            return root
        
        # 按文档中的出现顺序排序标题
        sorted_headers = sorted(headers, key=lambda x: doc['text'].find(x[0]))
        
        # 构建层次结构
        node_stack = [root]
        
        for i, (header_text, cluster) in enumerate(sorted_headers):
            # 确定层级
            granularity = self._determine_pdf_granularity(header_text)
            
            # 找到合适的父节点
            while len(node_stack) > 1 and node_stack[-1].granularity >= granularity:
                node_stack.pop()
            
            parent = node_stack[-1]
            
            # 计算文本位置
            start_pos = doc['text'].find(header_text)
            if start_pos == -1:
                start_pos = 0
            
            # 下一个标题的位置作为结束位置
            if i + 1 < len(sorted_headers):
                next_header = sorted_headers[i + 1][0]
                end_pos = doc['text'].find(next_header)
                if end_pos == -1:
                    end_pos = len(doc['text'])
            else:
                end_pos = len(doc['text'])
            
            # 提取该节点的上下文
            context = doc['text'][start_pos:end_pos]
            if self.need_clean_chunk:
                context = self._clean_text(context)
            
            # 创建节点
            node = SHTNode(
                node_id= i+1 ,  # f"{doc['id']}_node_{i}",
                doc_id=doc['doc_id'],
                name=header_text.strip(),
                granularity=granularity,
                context=context,
                summary="",
                start_pos=start_pos,
                end_pos=end_pos,
                parent=parent
            )
            parent.children.append(node)
            node_stack.append(node)
        # 合并全文context
        self._merge_full_context(root)
        return root


    def _build_tree_structure(self, headers, doc):
        """构建树形结构"""
        # 创建根节点
        root = SHTNode(
            node_id= 0 , # f"{doc['id']}_root",
            doc_id=doc['doc_id'],
            name=f"Document: {doc['name']}",
            granularity=0,
            context=doc['text'],  # 根节点-全文作为上下文。
            summary="",
            start_pos=0,
            end_pos=len(doc['text'])
        )
        
        if not headers:
            return root
        
        # 构建标题位置映射，处理重复标题问题
        header_positions = self._find_all_header_positions(headers, doc['text'])
        
        # 按实际位置排序标题
        sorted_headers = sorted(header_positions, key=lambda x: x[2])  # 按position排序
        
        # 构建层次结构
        node_stack = [root]
        
        for i, (header_text, cluster, start_pos) in enumerate(sorted_headers):
            # 确定层级
            granularity = self._determine_granularity(header_text)
            
            # 找到合适的父节点
            while len(node_stack) > 1 and node_stack[-1].granularity >= granularity:
                node_stack.pop()
            
            parent = node_stack[-1]
            
            # 下一个标题的位置作为结束位置
            if i + 1 < len(sorted_headers):
                end_pos = sorted_headers[i + 1][2]  # 使用下一个标题的实际位置
            else:
                end_pos = len(doc['text'])
            
            # 提取该节点的上下文
            context = doc['text'][start_pos:end_pos]
            if self.need_clean_chunk:
                context = self._clean_text(context)
            
            # 创建节点
            node = SHTNode(
                node_id= i+1 ,  # f"{doc['id']}_node_{i}", 根节点是0
                doc_id=doc['doc_id'],
                name=header_text.strip(),
                granularity=granularity,
                context=context,
                summary="",
                start_pos=start_pos,
                end_pos=end_pos,
                parent=parent
            )
            parent.children.append(node)
            node_stack.append(node)
        # 合并全文context
        self._merge_full_context(root)
        return root

    def _find_all_header_positions(self, headers, text):
        """
        找到所有标题在文本中的实际位置，处理重复标题问题
        
        Args:
            headers: [(header_text, cluster), ...]
            text: 文档全文
            
        Returns:
            [(header_text, cluster, position), ...] 按出现顺序排序
        """
        header_positions = []
        used_positions = set()  # 记录已使用的位置，避免重复
        
        for header_text, cluster in headers:
            # 找到该标题的所有出现位置
            start = 0
            while True:
                pos = text.find(header_text, start)
                if pos == -1:
                    break
                    
                # 如果这个位置还没被使用过，则添加
                if pos not in used_positions:
                    header_positions.append((header_text, cluster, pos))
                    used_positions.add(pos)
                    break  # 找到第一个未使用的位置就跳出
                    
                start = pos + 1  # 继续查找下一个位置
        
        return header_positions


    def _merge_full_context(self, node):
        """递归合并本节点及所有子节点的context到full_context"""
        if not node.children:
            node.full_context = node.context
        else:
            merged = node.context + '\n'
            for child in node.children:
                self._merge_full_context(child)
                merged += child.full_context + '\n'
            node.full_context = merged

    def _determine_granularity(self, header_text):
        header_match = re.match(r'^(#{1,6})\s+(.+)$', header_text )
        header_level = len(header_match.group(1))  # 获取#的数量
        return header_level

    def _determine_pdf_granularity(self, header_text):
        header = header_text.strip()
        # 匹配编号（如 1、1.1、1.1.1、1.1.1.1）
        m = re.match(r'^((\d+)(\.\d+)*)', header)
        if m:
            # 层级数 = 点的数量 + 1
            level = m.group(1).count('.') + 1
            return level
        elif header.isupper() and len(header) < 50:  # ABSTRACT 等全大写短标题
            return 1
        else:
            return 1  # 默认层级



    def _generate_summaries(self, node):
        """为节点生成摘要"""
        if not node:
            return
        
        # 生成节点摘要
        if node.context:
            # 提取关键句子作为摘要
            sentences = self._extract_key_sentences(node.context, max_sentences=MAX_SENTENCES)
            ancestors_path = " → ".join(node.get_ancestors())
            
            node.summary = f"""Title: {node.name}
Ancestors: {ancestors_path}
Key Content: {sentences}"""
        
        # 递归处理子节点
        for child in node.children:
            self._generate_summaries(child)
    
    def _extract_key_sentences(self, text, max_sentences=MAX_SENTENCES):
        """提取关键句子"""
        if not text:
            return ""        
        summary = tfidf_summary(text, max_sentences)
        return summary
    


    def _find_node_by_id(self, root, node_id):
        """根据ID找到节点"""
        if root.node_id == node_id:
            return root
        
        for child in root.children:
            result = self._find_node_by_id(child, node_id)
            if result:
                return result
        
        return None
    
    # 辅助模块
    def _llm_query(self, prompt, model="gpt-4.1", max_retries=3):
        """LLM调用封装"""
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=500
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"    ⚠️ LLM调用失败，重试 {attempt + 1}/{max_retries}: {e}")
                    time.sleep(1)
                else:
                    raise e
    
    # 主要工作流程
    def build_sht_for_all_docs(self, docs):
        """为所有文档构建SHT"""
        print("\n🏗️ 为所有文档构建语义层次树...")
        
        for doc in docs:
            sht_root = self._build_sht(doc)
            self.sht_tables[doc['doc_id']] = sht_root
            
        
        print(f"✓ 完成 {len(self.sht_tables)} 个文档的SHT构建")

    def get_docs_id(self) -> list[int]:
        """
        获取表中所有文档的doc_id

        Returns:
            list[int]: 文档ID列表
        """
        return list(self.docs_meta.keys())


    def _count_nodes(self, node):
        """计算节点总数"""
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count
    
    def _get_max_depth(self, node):
        """获取最大深度"""
        if not node.children:
            return node.granularity
        
        return max(self._get_max_depth(child) for child in node.children)

