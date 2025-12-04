from typing import List, Union
import numpy as np
from numpy.typing import NDArray
from langchain_core.embeddings import Embeddings
from transformers import AutoModel, AutoTokenizer
import torch
from tqdm import tqdm

from quest.conf.settings import LOCAL_MODEL_DIR

import os

E5_EMBEDDING_PATH = os.path.join(LOCAL_MODEL_DIR, "intfloat/multilingual-e5-large")

# ...existing code...

BGE_EMBEDDING_PATH = os.path.join(LOCAL_MODEL_DIR, "BAAI/bge-m3")

class batchedBGEEmbeddings(Embeddings):
    def __init__(self, model_path: str = BGE_EMBEDDING_PATH, device: str = "cuda", batch_size: int = 1):
        """初始化本地 BGE 嵌入模型
        
        Args:
            model_path: 本地模型路径 (如 '/path/to/bge-m3')
            device: 运行设备 ('cuda' 或 'cpu')
        """
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(device)
        self.emb_size = self.model.config.hidden_size
        self.device = device

    # overwrite   
    def embed_documents(self, texts: List[str]) -> List[NDArray]:
        """批量嵌入文档"""
        return self._embed(texts)
    
    # overwrite
    def embed_query(self, text: str) -> NDArray:
        """嵌入单个查询"""
        return self._embed([text])[0]
    
    def __call__(self, text: Union[str, List[str]]):
        """
        return : List[embedding] or embedding
        """
        if isinstance(text, str):
            return self.embed_query(text)
        return self.embed_documents(text)

    def _embed(self, sentences: List[str]) -> NDArray:
        """优化后的批量嵌入逻辑"""
        batch_size = self.batch_size
        all_embeddings = []
        
        # 分批次处理大规模输入
        for i in tqdm(range(0, len(sentences), batch_size), desc="BGE Embedding", unit="batch"):
            batch = sentences[i:i + batch_size]
            
            # 批量编码（自动处理填充和截断）
            inputs = self.tokenizer(
                batch, 
                return_tensors='pt', 
                truncation=True, 
                padding='longest',  # 动态填充到批次内最大长度
                max_length=8192
            ).to(self.device)  # 统一移动到指定设备
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # 提取[CLS]标记的嵌入（形状：[batch_size, hidden_size]）
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(cls_embeddings.cpu().numpy())
        
        # 合并所有批次的嵌入结果
        return np.concatenate(all_embeddings, axis=0)

# ...existing code...

class batchedE5Embeddings(Embeddings):
    def __init__(self, model_path: str = E5_EMBEDDING_PATH, device: str = "cuda", batch_size: int = 32):
        """初始化本地 E5 嵌入模型
        
        Args:
            model_path: 本地模型路径 (如 '/path/to/multilingual-e5-large')
            device: 运行设备 ('cuda' 或 'cpu')
        """
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(device)
        self.emb_size = self.model.config.hidden_size
        self.device = device

    # overwrite   
    def embed_documents(self, texts: List[str]) -> List[NDArray]:
        """批量嵌入文档"""
        return self._embed(texts)
    
    # overwrite
    def embed_query(self, text: str) -> NDArray:
        """嵌入单个查询"""
        return self._embed([text])[0]
    
    def __call__(self, text: Union[str, List[str]]):
        """
        return : List[embedding] or embedding
        """
        if isinstance(text, str):
            return self.embed_query(text)
        return self.embed_documents(text)

    def _embed(self, sentences: List[str]) -> NDArray:
        """优化后的批量嵌入逻辑"""
        batch_size = self.batch_size
        all_embeddings = []
        # 加入tqdm进度条显示

        
        # 分批次处理大规模输入
        for i in tqdm(range(0, len(sentences), batch_size), desc="E5 Embedding", unit="batch"):
        # for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            
            # 批量编码（自动处理填充和截断）
            inputs = self.tokenizer(
                batch, 
                return_tensors='pt', 
                truncation=True, 
                padding='longest',  # 动态填充到批次内最大长度
                max_length=512
            ).to(self.device)  # 统一移动到指定设备
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # 提取[CLS]标记的嵌入（形状：[batch_size, hidden_size]）
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(cls_embeddings.cpu().numpy())
        
        # 合并所有批次的嵌入结果
        return np.concatenate(all_embeddings, axis=0)


class E5Embeddings(Embeddings):
    def __init__(self, model_path: str = E5_EMBEDDING_PATH, device: str = "cuda"):
        """初始化本地 E5 嵌入模型
        
        Args:
            model_path: 本地模型路径 (如 '/path/to/multilingual-e5-large')
            device: 运行设备 ('cuda' 或 'cpu')
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(device)
        self.emb_size = self.model.config.hidden_size
        self.device = device

    # overwrite   
    def embed_documents(self, texts: List[str]) -> List[NDArray]:
        """批量嵌入文档"""
        return self._embed(texts)
    
    # overwrite
    def embed_query(self, text: str) -> NDArray:
        """嵌入单个查询"""
        return self._embed([text])[0]
    
    def __call__(self, text: Union[str, List[str]]):
        """
        return : List[embedding] or embedding
        """
        if isinstance(text, str):
            return self.embed_query(text)
        return self.embed_documents(text)

    def _embed(self, sentences: List[str]) -> List[NDArray]:
        """实际嵌入逻辑"""
        embeddings = []
        for x in sentences:
            # 对输入进行分词
            inputs = self.tokenizer(x, return_tensors='pt', truncation=True, padding=True, max_length=512)
            inputs = {k: v.to('cuda') for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            # 获取嵌入，E5 通常使用 [CLS] token 的嵌入
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
            embeddings.append(embedding)

        return np.array(embeddings)