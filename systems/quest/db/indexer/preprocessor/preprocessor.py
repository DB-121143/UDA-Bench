from typing import List, Union
import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod
from quest.core.datapack.doc import Doc
from quest.core.chunker.chunker import BaseChunker, GrammarSemanticChunker
from quest.core.embedding.e5Embedding import Embeddings, batchedE5Embeddings

class DocPreprocessor:
    def __init__(self, chunker: 'BaseChunker', embedding_model: 'Embeddings' = batchedE5Embeddings):
        """
        初始化文档预处理器

        Args:
            chunker: 文本分块器
            embedding_model: 嵌入模型
        """
        self.chunker = chunker
        self.embedding_model = embedding_model

    def emb_whole_document(self, doc: Doc) -> NDArray:
        return self.embedding_model([doc.content])[0]

    def preprocess_document(self, doc: Doc) -> tuple[List[str], List[NDArray]]:
        """
        预处理单个文档

        Args:
            doc: 文档对象

        Returns:
            tuple: (分块后的文本列表, 嵌入向量列表)
        """
        chunks = self.chunker.split_text(doc.content)
        embeddings = self.embedding_model(chunks)
        return chunks, embeddings

    def preprocess_documents(self, docs: List['Doc']) -> tuple[dict[int, List[str]], dict[int, List[NDArray]], dict[int, dict[str, any]], List[str]]:
        """
        批量预处理文档

        Args:
            docs: 文档列表

        Returns:
            tuple: (文档ID到分块列表的映射, 文档ID到嵌入列表的映射, 文档元数据, 所有唯一的元数据键列表)
        """
        doc2chunks = {}
        doc2embeddings = {}
        docs_meta = {}
        doc_2_whole_doc_embedding = {}

        for doc in docs:
            chunks, embeddings = self.preprocess_document(doc)
            doc2chunks[doc.doc_id] = chunks
            doc2embeddings[doc.doc_id] = embeddings
            docs_meta[doc.doc_id] = doc.metadata  # Doc 类有 metadata : dict属性,file_name就在其中
            doc_2_whole_doc_embedding[doc.doc_id] = self.emb_whole_document(doc)



        return doc2chunks, doc2embeddings, docs_meta, doc_2_whole_doc_embedding
