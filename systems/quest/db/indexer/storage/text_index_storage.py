"""VectorDBTextIndexStorage
基于OpenGauss数据库的向量存储实现，使用pgvector进行向量操作。
chunks和embeddings都存储在数据库表中，支持向量相似度查询。
"""
from __future__ import annotations
import json

from numpy.typing import NDArray
from typing import List, Tuple, Optional
import numpy as np
from sqlalchemy import text, MetaData, Table, Column, Integer, Text, Boolean
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.exc import SQLAlchemyError

from quest.db.connector.connector import create_opengauss_engine

from .index_storage import IndexStorage

def pgvector_to_numpy(embedding: str) -> NDArray:
    """将pgvector格式的字符串转换为numpy数组
    
    Args:
        embedding: pgvector格式的字符串，如 '[1.0,2.0,3.0]'
        
    Returns:
        numpy数组格式的向量
    """
    if isinstance(embedding, str):
        embedding_array = np.array(json.loads(embedding), dtype=np.float32)
    else:
        embedding_array = np.array(embedding, dtype=np.float32)
    return embedding_array

def numpy_to_pgvector(embedding: np.ndarray) -> str:
    """将numpy数组转换为pgvector格式的字符串
    
    Args:
        embedding: numpy数组格式的向量
        
    Returns:
        pgvector格式的字符串，如 '[1.0,2.0,3.0]'
    """
    return '[' + ','.join(map(str, embedding.tolist())) + ']'

class TextIndexStorage(IndexStorage):
    def __init__(self):
        super().__init__()
        pass

class VectorDBTextIndexStorage(TextIndexStorage):
    """基于OpenGauss数据库的向量存储实现
    
    将chunk文本和对应的embedding向量存储在数据库表中，
    使用pgvector语法进行向量相似度查询。
    """

    def __init__(self, table_name: str, db_conn=None, db_config=None, config_file=None, embedding_size = 128):
        """初始化VectorDBTextIndexStorage实例

        Args:
            table_name: 数据库表名
            db_conn: 数据库连接对象，如果为None则使用create_opengauss_engine创建
            db_config: 数据库配置字典，包含host, port, database, user, password
            config_file: 配置文件路径，如果提供则从文件读取配置
        """
        # 为分块级表添加_chunks后缀
        self.table_name = f"{table_name}_chunks"
        # table_name转化成小写
        self.table_name = self.table_name.lower()
        self.embedding_size = embedding_size

        # 数据库连接配置
        if db_conn is None:
            # 使用新的create_opengauss_engine方法创建数据库引擎
            self.engine = create_opengauss_engine(db_config=db_config, config_file=config_file)
        else:
            self.engine = db_conn

    def query_chunk_with_id_and_embedding(
        self,
        doc_id: Optional[int],
        topk: int,
        query_embedding: np.ndarray,
    ) -> List[Tuple[str, float, int, int, NDArray]]:
        """
        查询与输入最相似的chunks，并返回chunk_order和embedding

        Returns:
            [(chunk_text, similarity_score, doc_id, chunk_order, chunk_embedding), ...]
        """
        if query_embedding is None:
            raise ValueError("VectorDBTextIndexStorage.query_chunk_with_id_and_embedding requires query_embedding param")
        
        try:
            with self.engine.connect() as conn:
                query_embedding_vector = numpy_to_pgvector(query_embedding)
                
                if doc_id is not None:
                    sql = f"""
                    SELECT chunk_text, doc_id, chunk_order, embedding,
                        (1 - (embedding <=> :query_embedding)) as similarity
                    FROM {self.table_name}
                    WHERE doc_id = :doc_id
                    ORDER BY similarity DESC
                    """
                    params = {
                        'query_embedding': query_embedding_vector,
                        'doc_id': doc_id
                    }
                else:
                    sql = f"""
                    SELECT chunk_text, doc_id, chunk_order, embedding,
                        (1 - (embedding <=> :query_embedding)) as similarity
                    FROM {self.table_name}
                    ORDER BY similarity DESC
                    LIMIT :topk
                    """
                    params = {
                        'query_embedding': query_embedding_vector,
                        'topk': topk
                    }
                
                result = conn.execute(text(sql), params)
                candidates = []
                for row in result:
                    chunk_text, doc_id_result, chunk_order, embedding, similarity = row
                    # embedding 可能是字符串如 '[1.0,2.0,...]'，需转为 NDArray
                    embedding_array = pgvector_to_numpy(embedding)
                    candidates.append((chunk_text, float(similarity), int(doc_id_result), int(chunk_order), embedding_array))
                if doc_id is not None:
                    candidates = candidates[:topk]
                return candidates

        except SQLAlchemyError as e:
            raise RuntimeError(f"查询失败: {e}")



    def query_chunk_with_id(
        self,
        doc_id: Optional[int],
        topk: int,
        query_embedding: np.ndarray,
    ) -> List[Tuple[str, float, int, int]]:
        """
        查询与输入最相似的chunks，并返回chunk_order

        Returns:
            [(chunk_text, similarity_score, doc_id, chunk_order), ...]
        """
        if query_embedding is None:
            raise ValueError("VectorDBTextIndexStorage.query_chunk_with_id requires query_embedding param")
        
        try:
            with self.engine.connect() as conn:
                query_embedding_vector = numpy_to_pgvector(query_embedding)
                
                if doc_id is not None:
                    sql = f"""
                    SELECT chunk_text, doc_id, chunk_order,
                        (1 - (embedding <=> :query_embedding)) as similarity
                    FROM {self.table_name}
                    WHERE doc_id = :doc_id
                    ORDER BY similarity DESC
                    """
                    params = {
                        'query_embedding': query_embedding_vector,
                        'doc_id': doc_id
                    }
                else:
                    sql = f"""
                    SELECT chunk_text, doc_id, chunk_order,
                        (1 - (embedding <=> :query_embedding)) as similarity
                    FROM {self.table_name}
                    ORDER BY similarity DESC
                    LIMIT :topk
                    """
                    params = {
                        'query_embedding': query_embedding_vector,
                        'topk': topk
                    }
                
                result = conn.execute(text(sql), params)
                candidates = []
                for row in result:
                    chunk_text, doc_id_result, chunk_order, similarity = row
                    candidates.append((chunk_text, float(similarity), int(doc_id_result), int(chunk_order)))
                if doc_id is not None:
                    candidates = candidates[:topk]
                return candidates

        except SQLAlchemyError as e:
            raise RuntimeError(f"查询失败: {e}")



    def build_index(
        self,
        doc2chunks: dict[int, list[str]],
        doc2embeddings: dict[int, list[np.ndarray]],
    ) -> None:
        """构建索引，将chunks和embeddings存储到数据库中
        
        Args:
            doc2chunks: {doc_id: [chunk_text1, chunk_text2, ...]}
            doc2embeddings: {doc_id: [embedding1, embedding2, ...]}
        """
        try:
            with self.engine.connect() as conn:
                # 如果表存在，则删除已经存在的表
                drop_table_sql = f"DROP TABLE IF EXISTS {self.table_name};"
                conn.execute(text(drop_table_sql))

                # 创建表
                create_table_sql = f"""
                CREATE TABLE {self.table_name} (
                    id SERIAL PRIMARY KEY,
                    chunk_order INTEGER NOT NULL,
                    doc_id INTEGER NOT NULL,
                    chunk_text TEXT NOT NULL,
                    embedding vector({self.embedding_size}) NOT NULL
                );
                """
                conn.execute(text(create_table_sql))
                create_index_sql = f"CREATE INDEX idx_{self.table_name}_docid ON {self.table_name}(doc_id);"
                conn.execute(text(create_index_sql))                

                # 插入新数据
                for doc_id, chunks in doc2chunks.items():
                    embeddings = doc2embeddings[doc_id]
                    chunk_order = 1
                    
                    for chunk_text, embedding in zip(chunks, embeddings):
                        # 将numpy数组转换为pgvector格式
                        embedding_vector = numpy_to_pgvector(embedding)
                        
                        insert_sql = f"""
                        INSERT INTO {self.table_name} (chunk_order, doc_id,  chunk_text, embedding)
                        VALUES (:chunk_order, :doc_id, :chunk_text, :embedding)
                        """
                        
                        conn.execute(text(insert_sql), {
                            'chunk_order' : chunk_order,
                            'doc_id': doc_id,
                            'chunk_text': chunk_text,
                            'embedding': embedding_vector
                        })
                        chunk_order += 1
                
                conn.commit()
                
        except SQLAlchemyError as e:
            raise RuntimeError(f"数据库操作失败: {e}")

    def get_chunks_by_docid(self, doc_id: int) -> list[str]:
        """
        返回指定doc_id下的所有chunk_text，按照chunk_order从小到大排序

        Args:
            doc_id: 文档ID

        Returns:
            chunk_text组成的列表，顺序为chunk_order升序
        """
        try:
            with self.engine.connect() as conn:
                sql = f"""
                SELECT chunk_text FROM {self.table_name}
                WHERE doc_id = :doc_id
                ORDER BY chunk_order ASC
                """
                result = conn.execute(text(sql), {'doc_id': doc_id})
                return [row[0] for row in result]
        except SQLAlchemyError as e:
            raise RuntimeError(f"查询失败: {e}")


    def query(
        self,
        # query_text: str,
        doc_id: Optional[int],
        topk: int,
        query_embedding: np.ndarray,                
    ) -> List[Tuple[str, float, int]]:
        """查询与输入最相似的chunks
        
        Args:
            query_text: 查询文本（暂未使用）
            topk: 返回最相似的chunk数量
            doc_id: 如果指定，则只在该文档内查询
            query_embedding: 查询向量，必须提供
            
        Returns:
            [(chunk_text, similarity_score, doc_id), ...] 列表
        """
        if query_embedding is None:
            raise ValueError("VectorDBTextIndexStorage.query requires query_embedding param")
        
        try:
            with self.engine.connect() as conn:
                # 构建查询SQL
                query_embedding_vector = numpy_to_pgvector(query_embedding)
                
                if doc_id is not None:
                    # 限制在特定文档内查询 - 获取所有结果，在Python中处理TopK
                    sql = f"""
                    SELECT chunk_text, doc_id,
                           (1 - (embedding <=> :query_embedding)) as similarity
                    FROM {self.table_name}
                    WHERE doc_id = :doc_id
                    ORDER BY similarity DESC
                    """
                    # (1 - (embedding <=> :query_embedding::vector)) as similarity
                    params = {
                        'query_embedding': query_embedding_vector,
                        'doc_id': doc_id
                    }
                else:
                    # 在所有文档中查询
                    sql = f"""
                    SELECT chunk_text, doc_id,
                           (1 - (embedding <=> :query_embedding)) as similarity
                    FROM {self.table_name}
                    ORDER BY similarity DESC
                    LIMIT :topk
                    """
                    params = {
                        'query_embedding': query_embedding_vector,
                        'topk': topk
                    }
                
                result = conn.execute(text(sql), params)
                
                # 格式化返回结果
                candidates = []
                for row in result:
                    chunk_text, doc_id_result, similarity = row
                    candidates.append((chunk_text, float(similarity), int(doc_id_result)))
                
                # 如果指定了doc_id，在Python中处理TopK筛选
                if doc_id is not None:
                    candidates = candidates[:topk]
                
                return candidates
                
        except SQLAlchemyError as e:
            raise RuntimeError(f"查询失败: {e}")

    def save_index(self) -> None:
        """保存索引 - 对于数据库存储，此方法为空"""
        pass

    def load_index(self) -> bool:
        """检查数据库表是否存在
        
        Returns:
            如果表存在返回True，否则返回False
        """
        try:
            with self.engine.connect() as conn:
                check_table_sql = """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables 
                    WHERE table_name = :table_name
                );
                """
                result = conn.execute(text(check_table_sql), {'table_name': self.table_name})
                return result.scalar()
                
        except SQLAlchemyError:
            return False

