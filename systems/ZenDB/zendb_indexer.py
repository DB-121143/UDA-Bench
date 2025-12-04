#!/usr/bin/env python3
"""
ZenDB
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

#  API
DEEPSEEK_API_KEY = 'YOUR_API_KEY'
DEEPSEEK_BASE_URL = 'YOUR_BASE_URL'

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)

@dataclass
class SHTNode:
    """SHT Node"""
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
    full_context: str = None  # 
    
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
            # "full_context": self.full_context, # full_text
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
        """get_ancestors"""
        ancestors = []
        current = self.parent
        while current:
            ancestors.append(current.name)
            current = current.parent
        return ancestors[::-1]
    
    def is_leaf(self):
        """is leaf"""
        return len(self.children) == 0

from quest.conf.settings import ABS_PROJECT_ROOT_PATH, count_tokens

class ZenDBDocIndexer(SingleIndexer):
    """ZenDBIndexer"""
    
    def __init__(self, table_name: str, type: str = "ZenDBDoc", root_save_path = os.path.join(ABS_PROJECT_ROOT_PATH, "data/single_index"), need_clean_chunk = False, embedding_model = None, **kwargs):
        print("build ZenDB-singleIndex...")
        self.embedding_size = embedding_model.emb_size 
        self.embedding_model = embedding_model
        self.need_clean_chunk = need_clean_chunk
        self.table_name = table_name
        self.type = type

        self.table_save_path = os.path.join(root_save_path, f"{self.table_name}.json")
        self.embedding_save_path = os.path.join(root_save_path, f"{self.table_name}_embeddings.pkl")
        self.sht_tables = {}  # doc_id: int -> SHT nodes
        self.user_tables = {}  # table_name -> SQLite table
        self.templates = []    # save
        # new embedding vector
        self.node_embeddings: Dict[int, np.ndarray] = {}  # node_id -> embedding vector
        # self.embedding_cache: Dict[str, np.ndarray] = {}  # save the embedding
        self.query_embedding_cache: Dict[str, np.ndarray] = {}  # save same embedding
        # self.querier = OpenGaussQuerier()        

    def build_indexer(self, docs: List[ZenDBDoc]) -> None:
        # docs_meta
        self.docs_meta = {doc["doc_id"]: doc.metadata for doc in docs}
        # self.querier.build_cache_table(self.table_name, self.docs_meta)

        self.build_sht_for_all_docs(docs)
        # node embedding
        self._generate_embeddings_for_all_docs()
        self.save_indexer()
        return

    def save_indexer(self):
        # sht struct
        path = self.table_save_path
        sht_dict = {doc_id: root.to_dict() for doc_id, root in self.sht_tables.items()}
        data = {
            "docs_meta": self.docs_meta,
            "sht_trees": sht_dict
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # save embedding
        with open(self.embedding_save_path, "wb") as f:
            pickle.dump(self.node_embeddings, f)

    def load_indexer(self):
        # load SHT
        path = self.table_save_path
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.docs_meta = data["docs_meta"]
        # key to int
        self.docs_meta = {int(k): v for k, v in self.docs_meta.items()}
        self.sht_tables = {
            int(doc_id) : SHTNode.from_dict(root_dict)
            for doc_id, root_dict in data["sht_trees"].items()
        }
        
        # load embedding
        if os.path.exists(self.embedding_save_path):
            with open(self.embedding_save_path, "rb") as f:
                self.node_embeddings = pickle.load(f)
        return

    def _generate_embeddings_for_all_docs(self):
        """generate embedding"""
        print("\ngenerate embedding...")
        total_nodes = 0
        for doc_id, root in self.sht_tables.items():
            self._generate_embeddings(root)
            total_nodes += self._count_nodes(root)
        print(f"  ✓ done {total_nodes} nodes embedding")

    def _generate_embeddings(self, node: SHTNode) -> None:
        """embedding"""
        # get context node_id
        context_list, node_id_list = self.level_traverse(node.doc_id)
        
        if not context_list:
            return
        
        # embedding
        embeddings = self.embedding_model.embed_documents(context_list)
        
        # node_id to embedding
        for node_id, embedding in zip(node_id_list, embeddings):
            self.node_embeddings[node_id] = embedding

    def _get_embedding(self, node_id: int) -> np.ndarray:
        """get context embedding"""
        return self.node_embeddings.get(node_id, None)

    def _get_cached_query_embedding(self, query: str):
        """
        get cached query embedding，if not exist calc and cache
        
        Args:
            query: query string
            
        Returns:
            query embedding vector
        """
        cache_key = query[:256]  # use 256
        
        if cache_key not in self.query_embedding_cache:
            # calc embedding and cache
            self.query_embedding_cache[cache_key] = self.embedding_model.embed_query(query)
        
        return self.query_embedding_cache[cache_key]


    def _semantic_similarity_search(self, query: str, node_ids: List[int], topk: int) -> List[int]:
        """faiss optimize"""
        if not node_ids:
            return []
        
        query_embedding = self._get_cached_query_embedding(query)
        # query_embedding = self.embedding_model.embed_query(query)
        # embedding
        embeddings = []
        valid_node_ids = []
        for node_id in node_ids:
            emb = self._get_embedding(node_id)
            if emb is not None:
                embeddings.append(emb)
                valid_node_ids.append(node_id)
            else:
                raise ValueError(f"node {node_id} has no embedding")
        if not embeddings:
            return []
        # to np
        xb = np.array(embeddings).astype('float32')
        xq = np.array([query_embedding]).astype('float32')
        # cos
        faiss.normalize_L2(xb)
        faiss.normalize_L2(xq)
        # faiss
        index = faiss.IndexFlatIP(xb.shape[1])
        index.add(xb)
        D, I = index.search(xq, min(topk, xb.shape[0]))
        # node_id
        result_ids = [valid_node_ids[i] for i in I[0]]
        return result_ids

    def build_indexer(self, docs: List[ZenDBDoc]) -> None:
        # docs_meta
        self.docs_meta = {doc["doc_id"]: doc.metadata for doc in docs}
        # self.querier.build_cache_table(self.table_name, self.docs_meta)

        self.build_sht_for_all_docs(docs)
        # embedding
        self._generate_embeddings_for_all_docs()
        self.save_indexer()
        return

    def save_indexer(self):
        # save SHT
        path = self.table_save_path
        sht_dict = {doc_id: root.to_dict() for doc_id, root in self.sht_tables.items()}
        data = {
            "docs_meta": self.docs_meta,
            "sht_trees": sht_dict
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # save embedding
        with open(self.embedding_save_path, "wb") as f:
            pickle.dump(self.node_embeddings, f)

    def load_indexer(self):
        # load SHT
        path = self.table_save_path
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.docs_meta = data["docs_meta"]
        # key to int
        self.docs_meta = {int(k): v for k, v in self.docs_meta.items()}
        self.sht_tables = {
            int(doc_id) : SHTNode.from_dict(root_dict)
            for doc_id, root_dict in data["sht_trees"].items()
        }
        
        # embedding
        if os.path.exists(self.embedding_save_path):
            with open(self.embedding_save_path, "rb") as f:
                self.node_embeddings = pickle.load(f)
        return

    def get_file_name_by_id(self, doc_id: int) -> str:
        """
        ID to doc

        Args:
            doc_id

        Returns:
            str: filename
        """
        if doc_id in self.docs_meta:
            return self.docs_meta[doc_id].get("file_name", "")
        return ""

    def get_chunks_by_docid(self, doc_id):
        chunk_list, id_list = self.level_traverse(doc_id)
        return chunk_list

    def level_traverse(self, doc_id):
        """
        return  (context_list, node_id_list)
        """
        root = self.sht_tables.get(doc_id)
        if root is None:
            return [], []
        context_list = []
        node_id_list = []
        queue = [] 
        queue.extend(root.children) 
        while queue:
            node = queue.pop(0)
            context_list.append(node.context)
            node_id_list.append(node.node_id)
            queue.extend(node.children)
        return context_list, node_id_list


    def print_sht_tree(self, doc_id):
        """
        Args:
            doc_id
        """
        root = self.sht_tables.get(doc_id)
        if root is None:
            print(f"Doc: {doc_id} SHT not exist")
            return
        
        def _print_node(node, depth=0):
            indent = "  " * depth
            print(f"{indent}node_id: {node.node_id}")
            for child in node.children:
                _print_node(child, depth + 1)
        
        print(f"doc {doc_id} SHT:")
        _print_node(root)


    def llm_judge_contain_attr(self, node, query):
        try:
            # 
            prompt = (
                f"Determine if the following section contains the attribute defined in the query based on the summary of the section.\n"
                f"Query: {query}\n"
                f"Section Title: {node.name}\n"
                f"Section Summary: {node.summary}\n"
                f"Answer format: true or false, confidence from 0 to 100. e.g. (true, 100)"
            )

            input_tokens = count_tokens(prompt)
            response = self._llm_query(prompt)
            output_tokens = count_tokens(response)

            LLMInfo.add_input_tokens(input_tokens)
            LLMInfo.add_output_tokens(output_tokens)
            LLMInfo.add_query_times(1)

            m = re.search(r"(true|false)\s*[,，]\s*(\d{1,3})", response.lower(), re.IGNORECASE)
            if m:
                result = m.group(1).lower() == 'true'
                confidence = min(max(int(m.group(2)), 0), 100)
                return result, confidence
        except Exception:
            pass
        # default
        return False, 0

    def pre_order_dfs_search_related_node(self, node, query):
        if node is None:
            return False
        contain, confidence = self.llm_judge_contain_attr(node, query)
        if not contain:
            return False

        child_hit = False
        for child in node.children:
            if self.pre_order_dfs_search_related_node(child, query):
                child_hit = True

        if not child_hit:
            if not hasattr(self, 'candidate_nodes_with_confidence'):
                self.candidate_nodes = []
            self.candidate_nodes.append((node, confidence))
        return True


    def topk_semantic_beam_search_related_node(self, node: SHTNode, query: str, beam_topk: int) -> bool:
        self.candidate_nodes = []
        only_one_1_level_title_flag = False
        if beam_topk <= 0:
            return False
        
        
        # granularity=1
        # granularity_1_nodes = []
        # self._collect_nodes_by_granularity(node, 1, granularity_1_nodes)
        granularity_1_nodes = node.children
        if len(granularity_1_nodes) == 1:
            granularity_1_nodes = granularity_1_nodes[0].children
            only_one_1_level_title_flag = True
        
        if len(granularity_1_nodes) == 0 or not granularity_1_nodes:
            return False

        node_ids = [n.node_id for n in granularity_1_nodes]
        sorted_node_ids = self._semantic_similarity_search(query, node_ids, beam_topk)

        V1 = []  
        for node_id in sorted_node_ids:

            target_node = None
            for n in granularity_1_nodes:
                if n.node_id == node_id:
                    target_node = n
                    break
            
            if target_node:
                contain, confidence = self.llm_judge_contain_attr(target_node, query)
                if contain:
                    V1.append(target_node)
        

        all_candidate_nodes = V1.copy()  
        current_level_nodes = V1
        
        while current_level_nodes:
            next_level_nodes = []
            for parent_node in current_level_nodes:
                next_level_nodes.extend(parent_node.children)
            
            if not next_level_nodes:
                break
            
            # topK
            child_node_ids = [n.node_id for n in next_level_nodes]
            sorted_child_ids = self._semantic_similarity_search(query, child_node_ids, beam_topk)
            
            topk_children = []
            for node_id in sorted_child_ids:
                for child in next_level_nodes:
                    if child.node_id == node_id:
                        topk_children.append(child)
                        break
            
            all_candidate_nodes.extend(topk_children)
            current_level_nodes = topk_children
        
        # merge and get topK
        if all_candidate_nodes:
            unique_nodes = []
            seen_ids = set()
            for node in all_candidate_nodes:
                if node.node_id not in seen_ids:
                    unique_nodes.append(node)
                    seen_ids.add(node.node_id)
            
            final_node_ids = [n.node_id for n in unique_nodes]
            if only_one_1_level_title_flag:
                final_node_ids.append(granularity_1_nodes[0].node_id)
            final_sorted_ids = self._semantic_similarity_search(query, final_node_ids, beam_topk)
            
            for i, node_id in enumerate(final_sorted_ids[:beam_topk]):
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
        if node.granularity == target_granularity:
            result_list.append(node)
        
        for child in node.children:
            self._collect_nodes_by_granularity(child, target_granularity, result_list)


    def topk_query_all_nodes(self, doc_id: int, query: str, topk: int) -> List[tuple[str, int]]:
        """
        
        Args:
            doc_id:
            query:
            topk:
            
        Returns:
            List[tuple[str, int]]: [(node_context, node_id), ...] 
        """
        root = self.sht_tables.get(doc_id)
        if root is None:
            return []
        
        context_list, node_id_list = self.level_traverse(doc_id)
        
        if not node_id_list:
            return []
        
        sorted_node_ids = self._semantic_similarity_search(query, node_id_list, topk)

        results = []
        node_id_to_context = dict(zip(node_id_list, context_list))
        
        for node_id in sorted_node_ids:
            context = node_id_to_context.get(node_id, "")
            results.append((context, node_id))
        
        return results




    def get_relative_chunks_text_with_id(self, doc_id: int, query: str, topk: int) -> List[tuple[str, int]]:
        """
        return [(chunk_text, chunk_id), ...]
        """
        root = self.sht_tables.get(doc_id)
        if root is None:
            return []
        
        self.candidate_nodes = []
        
        has_candidate_flag = self.topk_semantic_beam_search_related_node(root, query, topk)    
        if not has_candidate_flag:
            return self.topk_query_all_nodes(doc_id, query, topk)    

        topk_nodes = self.candidate_nodes 
        
        results = []
        for node  in topk_nodes:
            results.append((node.context, node.node_id))
        
        
        return results

    def depreacated_topk_semantic_dfs_search_related_node(self, node: SHTNode, query: str, remaining_topk: int) -> bool:
        if remaining_topk <= 0:
            return False
        
        contain, confidence = self.llm_judge_contain_attr(node, query)
        
        if contain and confidence >= 70:
            if not hasattr(self, 'candidate_nodes_with_confidence'):
                self.candidate_nodes = []
            
            if len(self.candidate_nodes) < remaining_topk:
                self.candidate_nodes.append((node, confidence))
                remaining_topk -= 1
        
        if remaining_topk > 0 and node.children:
            child_ids = [child.node_id for child in node.children]
            sorted_child_ids = self._semantic_similarity_search(query, child_ids, len(child_ids))

            for child_id in sorted_child_ids:
                if remaining_topk <= 0:
                    break

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

            if not text:
                return ""
            
            text = re.sub(r'\(cid:\d+\)', '', text)
            
            text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)  # 
            text = re.sub(r'�', '', text)  # Unicode
            
            # 3. space
            text = re.sub(r'\s+', ' ', text)  # 
            text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # 
            
            # 4. 
            lines = text.split('\n')
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if line:  # 
                    cleaned_lines.append(line)
            
            text = '\n'.join(cleaned_lines)
            
            lines = text.split('\n')
            filtered_lines = []
            for line in lines:
                if re.match(r'^\s*[-\d\s]+\s*$', line) and len(line.strip()) < 5:
                    continue
                if re.match(r'^[\s\-_=~`!@#$%^&*(){}[\]|\\:";\'<>?,.]+$', line):
                    continue
                filtered_lines.append(line)
            
            text = '\n'.join(filtered_lines)
            
            # format
            text = text.strip()
            
            if len(re.findall(r'[a-zA-Z0-9]', text)) < 10:
                return ""
            
            return text


    def _extract_phrases(self, doc):
        def smart_split_phrase(phrase):
            m = re.match(r'^((?:\d+\.)*\d+)\s*([A-Za-z]+.*)', phrase)
            if m:
                prefix = m.group(1)
                rest = m.group(2)
                rest_split = ' '.join(wordninja.split(rest))
                return f"{prefix} {rest_split}"

            if re.match(r'^[A-Za-z]{10,}$', phrase):
                return ' '.join(wordninja.split(phrase))
            return phrase
            
        """vision merge"""
        words = doc["words"]
        if (not words) or  (doc["file_type"] == "txt") or (doc["file_type"] == "md"):
                # txt and md phrases
            line  = doc["text"]
            phrases = []
            lines = [line.strip() for line in doc["text"].splitlines() if line.strip()]

            normal_features = {
                "size": 12,  # 
                "fontname": "normal_text",
                "type": {"bold": False, "underline": False},
                "all_cap": line.isupper(),
                "num_st": line[:1].isdigit(),
                "alpha_st": line[:1].isalpha(),
                "center": False
            }

            title_features = {
                "size": 12,  # 
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
                # markdown
                header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
                
                if header_match:
                    # is markdown
                    header_level = len(header_match.group(1))  
                    header_text = line  
                    
                    # title fontname title{level}
                    features = title_features.copy()
                    features["fontname"] = f"title{header_level}"
                    features["type"]["bold"] = True
                    
                    # phrases
                    phrases.append((header_text, features))
                    i += 1
                else:
                    # not title
                    normal_text = line
                    normal_features_copy = normal_features.copy()
                    
                    j = i + 1
                    while j < len(lines):
                        next_line = lines[j]
                        if re.match(r'^(#{1,6})\s+(.+)$', next_line):
                            break  
                        normal_text += '\n' + next_line  
                        j += 1
                    
                    phrases.append((normal_text, normal_features_copy))
                    
                    i = j

            return phrases


        phrases = []
        current_phrase = ""
        current_features = None
        current_visual = None

        def get_word_visual(word):
            fontname = word.get("fontname", "")
            size = word.get("size", 12)
            is_bold = int("bold" in fontname.lower() or "Bold" in fontname)
            is_underline = int("underline" in fontname.lower())
            return (size, fontname, is_bold, is_underline)

        for i, word in enumerate(words):
            word_text = word.get("text", "")
            visual = get_word_visual(word)

            if current_phrase and visual != current_visual:
   
                phrase_text = current_phrase.strip()

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
                    "center": False  # 
                }
                phrases.append((phrase_text, features))
                current_phrase = word_text
                current_visual = visual
            else:
                current_phrase += ("" if current_phrase else "") + word_text
                current_visual = visual

 
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
       
        phrases = self._extract_phrases(doc)
        
        headers = self._identify_headers(phrases)
        
        root = self._build_tree_structure(headers, doc)
        
        self._generate_summaries(root)
        
        return root

    def _identify_headers(self, phrases):
        headers = []
        for phrase, features in phrases:
            if self._is_markdown_header(features):
                headers.append((phrase, features))

        return headers

    def _build_pdf_sht(self, doc):
        
        phrases = self._extract_phrases(doc)
        
        headers = self._identify_pdf_headers(phrases)
        
        root = self._build_pdf_tree_structure(headers, doc)
        
        self._generate_summaries(root)
        
        return root
    
    def _identify_pdf_headers(self, phrases, k=10, n_clusters=8):
        if not phrases:
            return []

        # phrases = [(phrase, info) for phrase, info in phrases if self._is_likely_header(phrase)]
        if not phrases:
            return []

        features = []
        for phrase, info in phrases:
            size = info['size']
            fontname = hash(info['fontname']) % 10000
            bold = int(info['type']['bold'])
            underline = int(info['type']['underline'])
            features.append([size, fontname, bold, underline])


        kmeans = KMeans(n_clusters=min(n_clusters, len(phrases)), random_state=42)
        cluster_ids = kmeans.fit_predict(features)


        for idx, (phrase, info) in enumerate(phrases):
            info['cluster'] = int(cluster_ids[idx])


        cluster_groups = {}
        for phrase, info in phrases:
            cluster = info['cluster']
            if cluster not in cluster_groups:
                cluster_groups[cluster] = []
            cluster_groups[cluster].append((phrase, info))

        headers = []
        headers_set = set()
        for cluster, cluster_phrases in cluster_groups.items():
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
                    #print(f"LLM: '{phrase}' -> {response}")
                    if response.strip().lower().startswith('y'):
                        header_votes += 1
                except Exception as e:
                    print(f"LLM error: {e}")
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
        """title"""
        phrase = phrase.strip()
        

        if len(phrase) < 5 or len(phrase) > 100:
            return False
            
        header_patterns = [
            r'^\d+\.?\s+',  # 
            r'^[A-Z][A-Z\s]+$',  # 
            r'(abstract|introduction|related work|methodology|experiments|conclusion)',  # 
            r'^(figure|table|algorithm)\s+\d+',  # 
        ]
        
        for pattern in header_patterns:
            if re.search(pattern, phrase, re.IGNORECASE):
                return True
                
        return False
    
    def _deprecated2_build_pdf_tree_structure(self, headers, doc):
        """tree"""
        root = SHTNode(
            node_id= 0 , # f"{doc['id']}_root",
            doc_id=doc['doc_id'],
            name=f"Document: {doc['name']}",
            granularity=0,
            context=doc['text'][:500],  # 
            summary="",
            start_pos=0,
            end_pos=len(doc['text'])
        )
        
        if not headers:
            return root
        
        sorted_headers = sorted(headers, key=lambda x: doc['text'].find(x[0]))
        
        node_stack = [root]
        
        for i, (header_text, cluster) in enumerate(sorted_headers):

            granularity = self._determine_pdf_granularity(header_text)
            
            while len(node_stack) > 1 and node_stack[-1].granularity >= granularity:
                node_stack.pop()
            
            parent = node_stack[-1]
            

            start_pos = doc['text'].find(header_text)
            if start_pos == -1:
                start_pos = 0
            
    
            if i + 1 < len(sorted_headers):
                next_header = sorted_headers[i + 1][0]
                end_pos = doc['text'].find(next_header)
                if end_pos == -1:
                    end_pos = len(doc['text'])
            else:
                end_pos = len(doc['text'])
            
            context = doc['text'][start_pos:end_pos]
            if self.need_clean_chunk:
                context = self._clean_text(context)
            
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
        # context
        self._merge_full_context(root)
        return root


    def _build_tree_structure(self, headers, doc):

        root = SHTNode(
            node_id= 0 , # f"{doc['id']}_root",
            doc_id=doc['doc_id'],
            name=f"Document: {doc['name']}",
            granularity=0,
            context=doc['text'],  # 
            summary="",
            start_pos=0,
            end_pos=len(doc['text'])
        )
        
        if not headers:
            return root
        
        header_positions = self._find_all_header_positions(headers, doc['text'])
        
        sorted_headers = sorted(header_positions, key=lambda x: x[2])  
        
        node_stack = [root]
        
        for i, (header_text, cluster, start_pos) in enumerate(sorted_headers):

            granularity = self._determine_granularity(header_text)
            
            while len(node_stack) > 1 and node_stack[-1].granularity >= granularity:
                node_stack.pop()
            
            parent = node_stack[-1]

            if i + 1 < len(sorted_headers):
                end_pos = sorted_headers[i + 1][2] 
            else:
                end_pos = len(doc['text'])
            
            context = doc['text'][start_pos:end_pos]
            if self.need_clean_chunk:
                context = self._clean_text(context)
            

            node = SHTNode(
                node_id= i+1 ,  # f"{doc['id']}_node_{i}", root 0
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
        # context
        self._merge_full_context(root)
        return root

    def _find_all_header_positions(self, headers, text):
        """
        
        Args:
            headers: [(header_text, cluster), ...]
            text: all doc
            
        Returns:
            [(header_text, cluster, position), ...] 
        """
        header_positions = []
        used_positions = set()  # 
        
        for header_text, cluster in headers:
            start = 0
            while True:
                pos = text.find(header_text, start)
                if pos == -1:
                    break
                    
                if pos not in used_positions:
                    header_positions.append((header_text, cluster, pos))
                    used_positions.add(pos)
                    break  # 
                    
                start = pos + 1  #
        
        return header_positions


    def _merge_full_context(self, node):
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
        header_level = len(header_match.group(1))  
        return header_level

    def _determine_pdf_granularity(self, header_text):
        header = header_text.strip()
        m = re.match(r'^((\d+)(\.\d+)*)', header)
        if m:
            level = m.group(1).count('.') + 1
            return level
        elif header.isupper() and len(header) < 50:  # ABSTRACT 
            return 1
        else:
            return 1  # 



    def _generate_summaries(self, node):
        if not node:
            return
        
        if node.context:
            sentences = self._extract_key_sentences(node.context, max_sentences=MAX_SENTENCES)
            ancestors_path = " → ".join(node.get_ancestors())
            
            node.summary = f"""Title: {node.name}
Ancestors: {ancestors_path}
Key Content: {sentences}"""
        
        for child in node.children:
            self._generate_summaries(child)
    
    def _extract_key_sentences(self, text, max_sentences=MAX_SENTENCES):
        if not text:
            return ""        
        summary = tfidf_summary(text, max_sentences)
        return summary
    


    def _find_node_by_id(self, root, node_id):
        if root.node_id == node_id:
            return root
        
        for child in root.children:
            result = self._find_node_by_id(child, node_id)
            if result:
                return result
        
        return None
    
    def _llm_query(self, prompt, model="gpt-4.1", max_retries=3):
        """LLM"""
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
                    print(f" LLM error {attempt + 1}/{max_retries}: {e}")
                    time.sleep(1)
                else:
                    raise e
    
    def build_sht_for_all_docs(self, docs):
        
        for doc in docs:
            sht_root = self._build_sht(doc)
            self.sht_tables[doc['doc_id']] = sht_root
            
        
        print(f" done {len(self.sht_tables)} doc SHT construct")

    def get_docs_id(self) -> list[int]:
        """
        doc_id

        Returns:
            list[int]: doc id list
        """
        return list(self.docs_meta.keys())


    def _count_nodes(self, node):
        """node sum"""
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count
    
    def _get_max_depth(self, node):
        """_get_max_depth"""
        if not node.children:
            return node.granularity
        
        return max(self._get_max_depth(child) for child in node.children)

