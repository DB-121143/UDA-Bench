"""Querier classes for database operations and cache management.

处理针对数据库的所有原生SQL语句操作，为DocIndexer的数据库访问提供支持。
为不同的数据库后端提供统一的上层操作接口。
基于sqlAlchemy core的SQL语句执行器统一封装。
使用opengauss数据库，兼容postgreSQL语法
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from sqlalchemy import text, MetaData, Table, Column, Integer, String, Text, Boolean, Float, DateTime
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql import select, insert, update, delete
import logging

from quest.db.connector.connector import create_opengauss_engine
from quest.db.indexer.storage.text_index_storage import numpy_to_pgvector, pgvector_to_numpy

logger = logging.getLogger(__name__)


class Querier(ABC):
    """数据库查询器抽象基类

    为不同的数据库后端提供统一的上层操作接口。
    处理针对数据库的所有原生SQL语句操作，为DocIndexer的数据库访问提供支持。
    """

    @abstractmethod
    def full_text_search_related_doc_ids(
        self,
        table_name: str,
        queries: List[str],
        threshold: float = 0.1
    ) -> List[List[int]]:
        """全文检索相关文档ID

        Args:
            table_name: 表名
            queries: 查询字符串列表
            threshold: 相似度阈值

        Returns:
            每个查询对应的文档ID列表 [[doc_id1, doc_id2], [doc_id3], ...]
        """
        pass


    @abstractmethod
    def build_cache_table(
        self,
        table_name: str,
        docs_meta: Dict[int, Dict[str, Any]],
        doc_2_whole_doc_embedding = None,
        doc_2_content: Dict[int, str] = None
    ) -> None:
        """建立缓存表

        Args:
            table_name: 表名
            docs_meta: 文档元数据，{doc_id: {attr: value, ...}}
            doc_2_whole_doc_embedding: 文档embedding映射
            doc_2_content: 文档内容映射，{doc_id: content_text}
        """
        pass

    @abstractmethod
    def load_docs_meta(
        self,
        table_name: str
    ) -> Dict[int, Dict[str, Any]]:
        """从数据库加载文档元数据

        Args:
            table_name: 表名

        Returns:
            文档元数据字典，{doc_id: {attr: value, ...}}
        """
        pass

    @abstractmethod
    def cache_doc_attr(
        self,
        table_name: str,
        doc_id: int,
        attr: str,
        value: Any,
        attr_type: str = "TEXT"
    ) -> None:
        """缓存指定文档的某个属性值

        Args:
            table_name: 表名
            doc_id: 文档ID
            attr: 属性名
            value: 属性值
            attr_type: 属性类型，用于动态创建列，默认为"TEXT"
        """
        pass

    @abstractmethod
    def get_doc_attr(
        self,
        table_name: str,
        doc_id: int,
        attr: str
    ) -> Any:
        """获取指定文档的某个属性值

        Args:
            table_name: 表名
            doc_id: 文档ID
            attr: 属性名

        Returns:
            属性值，类型取决于属性值类型
        """
        pass

    @abstractmethod
    def get_doc_attr_row(
        self,
        table_name: str,
        doc_id: int
    ) -> Dict[str, Any]:
        """获取指定文档的所有属性值

        Args:
            table_name: 表名
            doc_id: 文档ID

        Returns:
            属性名到属性值的映射 {attr: value, ...}
        """
        pass

    @abstractmethod
    def get_table_attr_column(
        self,
        table_name: str,
        attr: str
    ) -> Dict[int, Any]:
        """获取指定表中所有文档的某个属性值

        Args:
            table_name: 表名
            attr: 属性名

        Returns:
            文档ID到属性值的映射 {doc_id: value, ...}
        """
        pass


class OpenGaussQuerier(Querier):
    """使用OpenGauss作为存储后端的Querier实现

    基于SQLAlchemy Core进行SQL操作，兼容PostgreSQL语法。
    """

    def __init__(self, db_conn=None, db_config=None, config_file=None, embedding_size = 128):
        """初始化OpenGaussQuerier实例

        Args:
            db_conn: 数据库连接对象，如果为None则创建新连接
            db_config: 数据库配置字典，包含host, port, database, user, password
            config_file: 配置文件路径，如果提供则从文件读取配置
        """
        self.embedding_size = embedding_size
        if db_conn is None:
            # 使用connector模块创建数据库引擎
            self.engine = create_opengauss_engine(db_config=db_config, config_file=config_file)
        else:
            self.engine = db_conn

        self.metadata = MetaData()
        self._table_cache = {}  # 缓存已创建的表对象


    def _get_docs_table_name(self, table_name: str) -> str:
        """为文档级表添加_docs后缀

        Args:
            table_name: 原始表名

        Returns:
            添加_docs后缀的表名
        """
        # table_name转化成小写
        table_name = table_name.lower()
        return f"{table_name}_docs"

    def _get_sqlalchemy_type(self, value: Any):
        """根据Python值类型推断SQLAlchemy列类型"""
        if isinstance(value, bool):
            return Boolean
        elif isinstance(value, int):
            return Integer
        elif isinstance(value, float):
            return Float
        elif isinstance(value, str):
            # 对于较长的字符串使用Text类型
            if len(value) > 255:
                return Text
            else:
                return String(255)
        else:
            # 默认使用Text类型存储其他类型（如序列化后的对象）
            return Text

    def _drop_table_if_exists(self, table_name: str) -> None:
        """删除表（如果存在）"""
        # 获取实际的数据库表名（添加_docs后缀）
        actual_table_name = self._get_docs_table_name(table_name)

        try:
            with self.engine.connect() as conn:
                # 检查表是否存在
                check_table_sql = """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = :table_name
                );
                """
                result = conn.execute(text(check_table_sql), {'table_name': actual_table_name})
                table_exists = result.scalar()

                if table_exists:
                    # 删除表
                    drop_table_sql = f"DROP TABLE {actual_table_name}"
                    conn.execute(text(drop_table_sql))
                    conn.commit()
                    logger.info(f"Dropped existing table {actual_table_name}")

                    # 从缓存中移除表对象（使用实际表名）
                    if actual_table_name in self._table_cache:
                        del self._table_cache[actual_table_name]

                    # 从metadata中移除表定义（使用实际表名）
                    if actual_table_name in self.metadata.tables:
                        self.metadata.remove(self.metadata.tables[actual_table_name])

        except SQLAlchemyError as e:
            logger.error(f"Failed to drop table {actual_table_name}: {e}")
            raise RuntimeError(f"删除表失败: {e}")

    def _create_table_if_not_exists(
        self,
        table_name: str
    ) -> Table:
        """创建表（如果不存在）并返回Table对象"""
        # 获取实际的数据库表名（添加_docs后缀）
        actual_table_name = self._get_docs_table_name(table_name)

        if actual_table_name in self._table_cache:
            return self._table_cache[actual_table_name]

        try:
            with self.engine.connect() as conn:
                # 检查表是否已存在
                check_table_sql = """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = :table_name
                );
                """
                result = conn.execute(text(check_table_sql), {'table_name': actual_table_name})
                table_exists = result.scalar()

                if not table_exists:
                    # 创建包含doc_id、metadata和embedding列的表
                    create_table_sql = f"""
                    CREATE TABLE {actual_table_name} (
                        doc_id INTEGER PRIMARY KEY,
                        metadata JSONB,
                        embedding vector({self.embedding_size}) NOT NULL,
                        content TEXT
                    );
                    """
                    conn.execute(text(create_table_sql))
                    # 创建全文检索索引
                    create_fulltext_index_sql = f"""
                    CREATE INDEX idx_{actual_table_name}_fulltext 
                    ON {actual_table_name} 
                    USING gin(to_tsvector('english', content));
                    """
                    conn.execute(text(create_fulltext_index_sql))
                    

                    conn.commit()
                    logger.info(f"Created table {actual_table_name} with doc_id, metadata, and embedding columns")

                # 创建SQLAlchemy Table对象用于后续操作
                # 重新反射表结构
                self.metadata.reflect(bind=self.engine, only=[actual_table_name])
                table = self.metadata.tables[actual_table_name]
                self._table_cache[actual_table_name] = table
                return table

        except SQLAlchemyError as e:
            logger.error(f"Failed to create table {table_name}: {e}")
            raise RuntimeError(f"创建表失败: {e}")


    def build_cache_table(
        self,
        table_name: str,
        docs_meta: Dict[int, Dict[str, Any]],
        doc_2_whole_doc_embedding = None,
        doc_2_content: Dict[int, str] = None
    ) -> None:
        """建立缓存表，存储文档元数据、embedding和内容

        Args:
            table_name: 表名
            docs_meta: 文档元数据，{doc_id: {attr: value, ...}}
            doc_2_whole_doc_embedding: 文档embedding映射
            doc_2_content: 文档内容映射
        """
        actual_table_name = self._get_docs_table_name(table_name)

        try:
            # 先删除原有的同名表
            self._drop_table_if_exists(table_name)

            # 创建表
            self._create_table_if_not_exists(table_name)

            with self.engine.connect() as conn:
                # 插入文档元数据、embedding和内容
                for doc_id, doc_meta in docs_meta.items():
                    import json
                    metadata_json = json.dumps(doc_meta, ensure_ascii=False)
                    
                    # 获取embedding
                    embedding = None
                    if doc_2_whole_doc_embedding is not None:
                        embedding = doc_2_whole_doc_embedding.get(doc_id)
                        embedding = numpy_to_pgvector(embedding)
                    if embedding is None:
                        raise RuntimeError(f"缺少 doc_id={doc_id} 的 embedding，无法插入")

                    # 获取文档内容
                    content = ""
                    if doc_2_content is not None:
                        content = doc_2_content.get(doc_id, "")

                    insert_sql = f"""
                    INSERT INTO {actual_table_name} (doc_id, metadata, embedding, content)
                    VALUES (:doc_id, :metadata, :embedding, :content)
                    """

                    conn.execute(text(insert_sql), {
                        'doc_id': doc_id,
                        'metadata': metadata_json,
                        'embedding': embedding,
                        'content': content
                    })

                conn.commit()
                logger.info(f"Built cache table {actual_table_name} with {len(docs_meta)} documents")

        except SQLAlchemyError as e:
            logger.error(f"Failed to build cache table {actual_table_name}: {e}")
            raise RuntimeError(f"建立缓存表失败: {e}")
       

    def full_text_search_related_doc_ids(
        self,
        table_name: str,
        queries: List[str],
        threshold: float = 0.1
    ) -> List[List[int]]:
        """全文检索相关文档ID

        Args:
            table_name: 表名
            queries: 查询字符串列表
            threshold: 相似度阈值

        Returns:
            每个查询对应的文档ID列表
        """
        actual_table_name = self._get_docs_table_name(table_name)
        results = []

        try:
            with self.engine.connect() as conn:
                for query in queries:
                    # 使用PostgreSQL全文检索功能
                    sql = f"""
                    SELECT doc_id, 
                           ts_rank_cd(to_tsvector('english', content), 
                                     plainto_tsquery('english', :query)) as similarity
                    FROM {actual_table_name}
                    WHERE to_tsvector('english', content) @@ plainto_tsquery('english', :query)
                      AND ts_rank_cd(to_tsvector('english', content), 
                                    plainto_tsquery('english', :query)) >= :threshold
                    ORDER BY similarity DESC
                    """
                    
                    params = {
                        'query': query,
                        'threshold': threshold
                    }
                    
                    result = conn.execute(text(sql), params)
                    # doc_ids = [row[0] for row in result]
                    # similarities = [row[1] for row in result]
                    doc_id_with_similarities = [(row[0], row[1]) for row in result]
                    results.append(doc_id_with_similarities)
                    
                    # logger.debug(f"Full-text search for '{query}' found {len(doc_ids)} documents above threshold {threshold}")

            return results

        except SQLAlchemyError as e:
            logger.error(f"Failed to perform full-text search: {e}")
            raise RuntimeError(f"全文检索失败: {e}")




    def load_docs_meta(
        self,
        table_name: str
    ) -> Dict[int, Dict[str, Any]]:
        """从数据库加载文档元数据

        Args:
            table_name: 表名

        Returns:
            文档元数据字典，{doc_id: {attr: value, ...}}
        """
        # 获取实际的数据库表名（添加_docs后缀）
        actual_table_name = self._get_docs_table_name(table_name)

        try:
            with self.engine.connect() as conn:
                select_sql = f"SELECT doc_id, metadata FROM {actual_table_name}"
                result = conn.execute(text(select_sql))

                docs_meta = {}
                for row in result:
                    doc_id, metadata_json = row

                    # 反序列化JSON字符串
                    if metadata_json:
                        import json
                        try:
                            doc_meta = metadata_json #  json.loads(metadata_json)
                            docs_meta[doc_id] = doc_meta
                        except (json.JSONDecodeError, ValueError) as e:
                            logger.warning(f"Failed to parse metadata JSON for doc_id {doc_id}: {e}")
                            docs_meta[doc_id] = {}
                    else:
                        docs_meta[doc_id] = {}

                logger.info(f"Loaded metadata for {len(docs_meta)} documents from table {actual_table_name}")
                return docs_meta

        except SQLAlchemyError as e:
            logger.error(f"Failed to load docs meta from table {actual_table_name}: {e}")
            raise RuntimeError(f"加载文档元数据失败: {e}")

    def cache_doc_attr(
        self,
        table_name: str,
        doc_id: int,
        attr: str,
        value: Any,
        attr_type: str = "TEXT"
    ) -> None:
        """缓存指定文档的某个属性值

        Args:
            table_name: 表名
            doc_id: 文档ID
            attr: 属性名
            value: 属性值
            attr_type: 属性类型，用于动态创建列，默认为"TEXT"
        """
        # 获取实际的数据库表名（添加_docs后缀）
        actual_table_name = self._get_docs_table_name(table_name)

        try:
            with self.engine.connect() as conn:
                # 检查表是否存在
                check_table_sql = """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = :table_name
                );
                """
                result = conn.execute(text(check_table_sql), {'table_name': actual_table_name})
                table_exists = result.scalar()

                if not table_exists:
                    raise RuntimeError(f"表 {actual_table_name} 不存在，请先调用 build_cache_table 创建表")

                # 检查列是否存在
                check_column_sql = """
                SELECT EXISTS (
                    SELECT 1  FROM information_schema.columns
                    WHERE table_name = :table_name AND column_name = :column_name
                );
                """
                result = conn.execute(text(check_column_sql), {'table_name': actual_table_name, 'column_name': attr})
                column_exists = result.scalar()

                # 如果列不存在，则添加列
                if not column_exists:
                    add_column_sql = f"ALTER TABLE {actual_table_name} ADD COLUMN {attr} {attr_type}"
                    conn.execute(text(add_column_sql))
                    logger.info(f"Added column {attr} with type {attr_type} to table {actual_table_name}")

                # 处理复杂类型
                if isinstance(value, (dict, list)):
                    import json
                    value = json.dumps(value, ensure_ascii=False)

                # 检查文档是否存在
                check_sql = f"SELECT COUNT(*) FROM {actual_table_name} WHERE doc_id = :doc_id"
                result = conn.execute(text(check_sql), {'doc_id': doc_id})
                exists = result.scalar() > 0

                if exists:
                    # 更新现有记录
                    update_sql = f"UPDATE {actual_table_name} SET {attr} = :value WHERE doc_id = :doc_id"
                    conn.execute(text(update_sql), {'value': value, 'doc_id': doc_id})
                else:
                    # 插入新记录
                    insert_sql = f"INSERT INTO {actual_table_name} (doc_id, {attr}) VALUES (:doc_id, :value)"
                    conn.execute(text(insert_sql), {'doc_id': doc_id, 'value': value})

                conn.commit()
                logger.debug(f"Cached attribute {attr} for doc {doc_id} in table {actual_table_name}")

        except SQLAlchemyError as e:
            logger.error(f"Failed to cache doc attribute: {e}")
            raise RuntimeError(f"缓存文档属性失败: {e}")

    def get_doc_attr(
        self,
        table_name: str,
        doc_id: int,
        attr: str
    ) -> Any:
        """获取指定文档的某个属性值

        Args:
            table_name: 表名
            doc_id: 文档ID
            attr: 属性名

        Returns:
            属性值，类型取决于属性值类型
        """
        # 获取实际的数据库表名（添加_docs后缀）
        actual_table_name = self._get_docs_table_name(table_name)

        try:
            with self.engine.connect() as conn:
                select_sql = f"SELECT {attr} FROM {actual_table_name} WHERE doc_id = :doc_id"
                result = conn.execute(text(select_sql), {'doc_id': doc_id})
                row = result.fetchone()

                if row is None:
                    return None

                value = row[0]

                # 尝试反序列化JSON字符串
                if isinstance(value, str):
                    try:
                        import json
                        return json.loads(value)
                    except (json.JSONDecodeError, ValueError):
                        # 如果不是JSON字符串，直接返回原值
                        return value

                return value

        except SQLAlchemyError as e:
            logger.error(f"Failed to get doc attribute: {e}")
            # raise RuntimeError(f"获取文档属性失败: {e}")
            return None

    def get_doc_attr_row(
        self,
        table_name: str,
        doc_id: int
    ) -> Dict[str, Any]:
        """获取指定文档的所有属性值

        Args:
            table_name: 表名
            doc_id: 文档ID

        Returns:
            属性名到属性值的映射 {attr: value, ...}
        """
        # 获取实际的数据库表名（添加_docs后缀）
        actual_table_name = self._get_docs_table_name(table_name)

        try:
            with self.engine.connect() as conn:
                select_sql = f"SELECT * FROM {actual_table_name} WHERE doc_id = :doc_id"
                result = conn.execute(text(select_sql), {'doc_id': doc_id})
                row = result.fetchone()

                if row is None:
                    return {}

                # 获取列名
                columns = result.keys()
                row_dict = dict(zip(columns, row))

                # 移除doc_id列，只返回属性
                if 'doc_id' in row_dict:
                    del row_dict['doc_id']

                # 尝试反序列化JSON字符串
                for key, value in row_dict.items():
                    if isinstance(value, str):
                        try:
                            import json
                            row_dict[key] = json.loads(value)
                        except (json.JSONDecodeError, ValueError):
                            # 如果不是JSON字符串，保持原值
                            pass

                return row_dict

        except SQLAlchemyError as e:
            logger.error(f"Failed to get doc attribute row: {e}")
            raise RuntimeError(f"获取文档属性行失败: {e}")

    def get_table_attr_column(
        self,
        table_name: str,
        attr: str
    ) -> Dict[int, Any]:
        """获取指定表中所有文档的某个属性值

        Args:
            table_name: 表名
            attr: 属性名

        Returns:
            文档ID到属性值的映射 {doc_id: value, ...}
        """
        # 获取实际的数据库表名（添加_docs后缀）
        actual_table_name = self._get_docs_table_name(table_name)

        try:
            with self.engine.connect() as conn:
                select_sql = f"SELECT doc_id, {attr} FROM {actual_table_name}"
                result = conn.execute(text(select_sql))

                attr_dict = {}
                for row in result:
                    doc_id, value = row

                    # 尝试反序列化JSON字符串
                    if isinstance(value, str):
                        try:
                            import json
                            value = json.loads(value)
                        except (json.JSONDecodeError, ValueError):
                            # 如果不是JSON字符串，保持原值
                            pass

                    attr_dict[doc_id] = value

                return attr_dict

        except SQLAlchemyError as e:
            logger.error(f"Failed to get table attribute column: {e}")
            raise RuntimeError(f"获取表属性列失败: {e}")

    def load_docs_embedding(
        self,
        table_name: str
    ) -> Dict[int, Any]:
        """从数据库加载所有文档的embedding

        Args:
            table_name: 表名

        Returns:
            doc_id 到 embedding 的映射字典 {doc_id: embedding, ...}
        """
        actual_table_name = self._get_docs_table_name(table_name)
        try:
            with self.engine.connect() as conn:
                select_sql = f"SELECT doc_id, embedding FROM {actual_table_name}"
                result = conn.execute(text(select_sql))
                doc_2_whole_doc_embedding = {}
                for row in result:
                    doc_id, embedding = row
                    # embedding要转化为NDArray的形式：
                    embedding = pgvector_to_numpy(embedding)
                    doc_2_whole_doc_embedding[doc_id] = embedding
                return doc_2_whole_doc_embedding
        except SQLAlchemyError as e:
            logger.error(f"Failed to load docs embedding from table {actual_table_name}: {e}")
            raise RuntimeError(f"加载文档embedding失败: {e}")