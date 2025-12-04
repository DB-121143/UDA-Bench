"""
索引存储基类模块
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Any
import numpy as np

class IndexStorage(ABC):
    """
    实现索引存储和查询的最底层工具类基类
    """
    
    @abstractmethod
    def build_index(self, doc2chunks: dict[int, list[str]], doc2embeddings: dict[int, list[np.ndarray]]):
        """构建索引"""
        pass


