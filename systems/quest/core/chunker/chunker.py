from langchain_core.documents import Document
from typing import List, Dict, Optional
from abc import ABC, abstractmethod
from quest.core.datapack.doc import Doc

from quest.conf.settings import ABS_PROJECT_ROOT_PATH
import os
from quest.conf.settings import Enc_token_cnt

# 基于Character的TextSplitter，最小单元是单个字符，比如'a'
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

# 基于Token的TextSplitter，最小单元是单个Token,比如'hello'
from  langchain.text_splitter import SentenceTransformersTokenTextSplitter

# 基于NLTK的TextSplitter，最小单元是??
from  langchain.text_splitter import NLTKTextSplitter

from langchain_experimental.text_splitter import SemanticChunker

class BaseChunker(ABC):
    
    @abstractmethod
    def split_text(self, text : str) -> List[str]:
        pass
  
    def chunk_documents(self, docs_list : List[Doc]) -> Dict[int, List[str]]:
        """
        文档预处理流水线
        :param docs_list: Doc对象列表
        :return: 字典，键为文档ID，值为该文档对应的文本块列表
        """
        doc2chunks = {}
        
        for doc in docs_list:
            text = doc.content
            chunks = self.split_text(text)
            doc2chunks[doc.doc_id] = chunks
            
        return doc2chunks        

class TokenTextChunker(BaseChunker):
    """
    按token数分块，支持重叠
    """
    def __init__(self, chunk_size: int, chunk_overlap: int):
        super().__init__()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:

        tokens = Enc_token_cnt.encode(text)
        
        chunks = []
        start = 0
        
        while start < len(tokens):
            # 计算当前块的结束位置
            end = min(start + self.chunk_size, len(tokens))
            
            # 提取当前块的tokens并解码为文本
            chunk_tokens = tokens[start:end]
            chunk_text = Enc_token_cnt.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            # 如果到达文本末尾，直接退出
            if end == len(tokens):
                break

            # 计算下一块的起始位置，考虑重叠
            start = end - self.chunk_overlap
            
        
        return chunks


from .splitter import GrammarSemanticSplitter, RecursiveTokenTextSplitter

class SentenceTransformerTokenTextChunker(BaseChunker):
    """
    把所有文档的分块全存储在内存中
    """
    def __init__(self, 
                 tokens_per_chunk: int = 356, 
                 chunk_overlap: int = 100,
                 model_name = os.path.join(ABS_PROJECT_ROOT_PATH, "model/sentence-transformers/all-mpnet-base-v2")
                 ):
        
        self.text_splitter = SentenceTransformersTokenTextSplitter(
            chunk_overlap = chunk_overlap,
            tokens_per_chunk = tokens_per_chunk,
            model_name=model_name
            )
        
    def split_text(self, text):
        return self.text_splitter.split_text(text)

class GrammarSemanticChunker(BaseChunker):
    def __init__(
        self,
        embeddings ,
        buffer_size: int = 1,
        min_chunk_size: Optional[int]= None,
        breakpoint_threshold_type = "percentile",
        breakpoint_threshold_amount: Optional[float] = None,
        language: str = "english",
        use_span_tokenize: bool = False, # 是否保留分句后的空白符。
        number_of_chunks = None,    
        max_chunk_size: Optional[int] = None    
    ):
        self.text_splitter = GrammarSemanticSplitter(
            embeddings=embeddings,
            buffer_size=buffer_size,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
            language=language,
            use_span_tokenize=use_span_tokenize,
            number_of_chunks = number_of_chunks,
        )

    def split_text(self, text):
        return self.text_splitter.split_text(text) 


class NLTKTokenTextChunker(BaseChunker):
    """
    把所有文档的分块全存储在内存中
    """
    def __init__(self, 
                 separator: str = "\n\n",
                 language: str = "english",
                 use_span_tokenize: bool = False
                 ):
        
        self.text_splitter = NLTKTextSplitter(
            separator=separator,
            language=language,
            use_span_tokenize=use_span_tokenize
        )

    def split_text(self, text):
        return self.text_splitter.split_text(text)

class SemanticTextChunker(BaseChunker):
    """
    把所有文档的分块全存储在内存中
    """
    def __init__(self, model, min_chunk_size=5, number_of_chunks=None):     
        self.text_splitter = SemanticChunker(model, min_chunk_size=min_chunk_size, number_of_chunks=number_of_chunks)

    def split_text(self, text):
        return self.text_splitter.split_text(text)        

class RecursiveCharacterTextChunker(BaseChunker):
    """
    把所有文档的分块全存储在内存中
    """
    def __init__(self, 
                 chunk_size: int = 512, 
                 chunk_overlap: int = 128,):
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True
        )

    def split_text(self, text):
        return self.text_splitter.split_text(text)



class RecursiveTokenTextChunker(BaseChunker):
    """
    把所有文档的分块全存储在内存中
    """
    def __init__(self, 
                 
                 chunk_size: int = 512, 
                 chunk_overlap: int = 128,   
                 separators: Optional[List[str]] = ["\n\n", ".", "?", "!", ";" ,"\n", " ", ""],              
                 ):
        
        self.text_splitter = RecursiveTokenTextSplitter(
            separators = separators,
            chunk_size=chunk_size,  # 现在是 token 数量而不是字符数量
            chunk_overlap=chunk_overlap            
        )

    def split_text(self, text):
        return self.text_splitter.split_text(text)

