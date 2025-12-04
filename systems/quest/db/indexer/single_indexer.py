from typing import List
from quest.db.indexer.preprocessor.preprocessor import DocPreprocessor
from quest.db.indexer.storage.text_index_storage import VectorDBTextIndexStorage
from quest.db.querier.querier import OpenGaussQuerier
from quest.core.datapack.doc import Doc
from quest.core.chunker.chunker import RecursiveCharacterTextChunker, GrammarSemanticChunker, SentenceTransformerTokenTextChunker
from quest.core.embedding.e5Embedding import batchedE5Embeddings
from quest.conf import settings
from numpy.typing import NDArray

class SingleIndexer:
    """
    单表索引器类，负责单个表的索引操作
    """

    def __init__(self, table_name: str, type: str, chunker = None, **kwargs):
        """
        初始化SingleIndexer实例

        Args:
            table_name: 表名，用于指定索引文件路径/数据库表名
            type: 索引类型，表示文档的类型，比如"TextDoc"、"PdfDoc"等。
            **kwargs: 其他参数
        """
        self.table_name = table_name
        self.type = type
        self.docs_meta = {}
        self.preprocessor = None
        self.querier = None

        def get_file_name_by_id(self, doc_id: int) -> str:
            pass

        def get_docs_id(self) -> list[int]:
            pass

        def get_relative_chunks_text(self, doc_id: int, query: str, topk: int) -> List[str]:
            pass


class TextDocIndexer(SingleIndexer):
    def __init__(self, table_name: str, type: str = "TextDoc", chunker = None, embedding_model = None, **kwargs):
        super().__init__(table_name, type, **kwargs)
        
        self.embedding_size = embedding_model.emb_size     
        self.querier = OpenGaussQuerier(embedding_size=self.embedding_size)

        # Initialize preprocessor with chunker and embedding model
        self.storage = VectorDBTextIndexStorage(table_name, embedding_size=self.embedding_size)
        
        self.preprocessor = DocPreprocessor(chunker=chunker, embedding_model=embedding_model)
        
        # 添加query embedding缓存
        self.query_embedding_cache = {}

    def get_chunks_by_docid(self, doc_id) -> list[str]:
        return self.storage.get_chunks_by_docid(doc_id)

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


    def get_docs_id(self) -> list[int]:
        """
        获取表中所有文档的doc_id

        Returns:
            list[int]: 文档ID列表
        """
        return list(self.docs_meta.keys())

    def build_indexer(self, docs: List[Doc]) -> None:
        """
        构建文档索引

        Args:
            docs: 文档列表
        """
        # 1. 使用预处理器处理文档，获取chunks、embeddings和metadata
        doc2chunks, doc2embeddings, docs_meta, doc_2_whole_doc_embedding = self.preprocessor.preprocess_documents(docs)

        # 2. 存储文档元数据 和 embedding，embedding后续可以用来聚类。
        self.docs_meta = docs_meta
        self.doc_2_whole_doc_embedding = doc_2_whole_doc_embedding

        # 3. 构建文档内容映射（用于全文检索）
        doc_2_content = {}
        for doc in docs:
            doc_id = doc.doc_id
            if hasattr(doc, 'content') and doc.content:
                doc_2_content[doc_id] = doc.content       

        # 3. 调用storage构建向量索引
        self.storage.build_index(doc2chunks, doc2embeddings)

        # 4. 使用querier建立缓存表，存储文档元数据到数据库
        # jztodo 像文档缓存表里额外存储一个doc的embedding字段。
        self.querier.build_cache_table(self.table_name, docs_meta, doc_2_whole_doc_embedding, doc_2_content)

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
            self.query_embedding_cache[cache_key] = self.preprocessor.embedding_model.embed_query(query)
        
        return self.query_embedding_cache[cache_key]

    def get_relative_chunks_text_with_id_and_embedding(self, doc_id: int, query: str, topk: int) -> List[tuple[str, int, NDArray]]:
        """
        查询指定文档中与query相关的文本块内容

        Args:
            doc_id: 文档编号
            query: 查询字符串
            topk: 返回的相关块数量

        Returns:
            与query相关的topk个块内容列表及其chunk_id， List[chunk_text, chunk_id]
            chunk_id在文档范围内是唯一的，不同的文档中的块可能有相同的chunk_id
        """
        # 1. 使用缓存的embedding模型将查询转换为向量
        query_embedding = self._get_cached_query_embedding(query)

        # 2. 调用storage查询相似的文本块
        results = self.storage.query_chunk_with_id_and_embedding(
            doc_id=doc_id,
            topk=topk,
            query_embedding=query_embedding
        )

        # 3. 提取文本块内容并返回
        chunk_texts_with_id_and_embedding = [(result[0], result[3], result[4]) for result in results]  # result格式: (chunk_text, similarity_score, doc_id, chunk_id, chunk_embedding)
        return chunk_texts_with_id_and_embedding # (chunk_text, chunk_id, chunk_embedding)


    def full_text_search_related_docs(self, queries: List[str], threshold: float = 0.1) -> List[List[int]]:
        """
        全文检索相关文档

        Args:
            queries: 查询字符串列表
            threshold: 相似度阈值

        Returns:
            每个查询对应的文档ID列表
        """
        if hasattr(self, 'querier') and self.querier:
            return self.querier.full_text_search_related_doc_ids(
                self.table_name, 
                queries, 
                threshold
            )
        else:
            return [[] for _ in queries]



    def get_relative_chunks_text_with_id(self, doc_id: int, query: str, topk: int) -> List[tuple[str, int]]:
        """
        查询指定文档中与query相关的文本块内容

        Args:
            doc_id: 文档编号
            query: 查询字符串
            topk: 返回的相关块数量

        Returns:
            与query相关的topk个块内容列表及其chunk_id， List[chunk_text, chunk_id]
            chunk_id在文档范围内是唯一的，不同的文档中的块可能有相同的chunk_id
        """
        # 1. 使用缓存的embedding模型将查询转换为向量
        query_embedding = self._get_cached_query_embedding(query)

        # 2. 调用storage查询相似的文本块
        results = self.storage.query_chunk_with_id(
            doc_id=doc_id,
            topk=topk,
            query_embedding=query_embedding
        )

        # 3. 提取文本块内容并返回
        chunk_texts_with_id = [(result[0], result[3]) for result in results]  # result格式: (chunk_text, similarity_score, doc_id, chunk_id)
        return chunk_texts_with_id # (chunk_text, chunk_id)


    def get_relative_chunks_text(self, doc_id: int, query: str, topk: int) -> List[str]:
        """
        查询指定文档中与query相关的文本块内容

        Args:
            doc_id: 文档编号
            query: 查询字符串
            topk: 返回的相关块数量

        Returns:
            与query相关的topk个块内容列表
        """
        # 1. 使用缓存的embedding模型将查询转换为向量
        query_embedding = self._get_cached_query_embedding(query)

        # 2. 调用storage查询相似的文本块
        results = self.storage.query(
            doc_id=doc_id,
            topk=topk,
            query_embedding=query_embedding
        )

        # 3. 提取文本块内容并返回
        chunk_texts = [result[0] for result in results]  # result格式: (chunk_text, similarity_score, doc_id)
        return chunk_texts
    
    def get_relative_chunks_lenght(self, doc_id_list: List[int], query: str, topk: int) -> int:
        """
        查询指定文档中与query相关的文本块长度

        Args:
            doc_id_list: 文档编号List
            query: 查询字符串
            topk: 返回的相关块数量

        Returns:
            与query相关的doc_id_list
        
        """
        # 1. 使用缓存的embedding模型将查询转换为向量
        query_embedding = self._get_cached_query_embedding(query)

        # 2. 调用storage查询相似的文本块

        res = 0
        for id in doc_id_list:
            results = self.storage.query(
                doc_id=id,
                topk=topk,
                query_embedding=query_embedding
            )

            # 3. 提取文本块token数并返回
            for result in results:
                res += len(settings.enc.encode(result[0]))
            
        return res

    def save_indexer(self) -> None:
        """
        保存索引到磁盘
        """
        self.storage.save_index()

    def get_doc_embedding(self, doc_id)  -> NDArray :
        return self.doc_2_whole_doc_embedding[doc_id]

    def load_indexer(self) -> None:
        """
        从磁盘加载索引
        """
        # 1. 从数据库加载文档元数据
        self.docs_meta = self.querier.load_docs_meta(self.table_name)
        self.doc_2_whole_doc_embedding = self.querier.load_docs_embedding(self.table_name)

        # 2. 加载向量索引
        load_flag = False
        load_flag = self.storage.load_index()
        if load_flag:
            print(f"成功加载索引: {self.table_name}")
        else:
            print(f"索引不存在: {self.table_name}")
            raise ValueError(f"索引不存在: {self.table_name}")


class HierarchicalTextDocIndexer(SingleIndexer):
    def __init__():
        pass



def bottom():
    pass

