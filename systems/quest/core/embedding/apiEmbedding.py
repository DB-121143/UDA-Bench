from typing import List, Union
import numpy as np
from numpy.typing import NDArray
from langchain_core.embeddings import Embeddings
from tqdm import tqdm
import litellm  # 确保你已安装 litellm
from quest.conf.settings import count_tokens

class ApiEmbeddings(Embeddings):
    def __init__(
        self, 
        model: str = "openai/Qwen/Qwen3-Embedding-0.6B",  # 你可以自定义默认模型
        api_base: str = None,   # 你如果用本地 api base 可以传入
        api_key: str = None,
        max_emb_size: int = 8190,
        batch_size: int = 32
    ):
        """
        Args:
            model: litellm 支持的模型名（如 OpenAI、Qwen、Mistral 等 Embedding 模型）
            api_base: 可选，api base 地址（如用本地部署 Embedding 服务）
            api_key: 可选，API 密钥
        """
        self.max_emb_size = max_emb_size
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.batch_size = batch_size

        # 调用一次 embedding 接口以获得 emb_size
        test_emb = self._embed(["test"])
        self.emb_size = test_emb.shape[1] if len(test_emb.shape) == 2 else len(test_emb)

    def embed_documents(self, texts: List[str]) -> List[NDArray]:
        """批量嵌入文档"""
        return self._embed(texts)

    def embed_query(self, text: str) -> NDArray:
        """嵌入单个查询"""
        return self._embed([text])[0]

    def __call__(self, text: Union[str, List[str]]):
        if isinstance(text, str):
            return self.embed_query(text)
        return self.embed_documents(text)

    def _embed(self, sentences: List[str]) -> NDArray:
        all_embeddings = []
        batch_size = self.batch_size

        # litellm 通常只支持按模型规定的最大 batch
        for i in tqdm(range(0, len(sentences), batch_size), desc="LiteLLM Embedding", unit="batch"):
            batch = sentences[i:i+batch_size]
            batch = [sentence[:self.max_emb_size] if count_tokens(sentence) > self.max_emb_size else sentence for sentence in batch]

            response = litellm.embedding(
                model=self.model,
                input=batch,
                api_base=self.api_base,
                api_key=self.api_key
            )
            # response["data"] 是 embedding 结果
            embeddings = [item["embedding"] for item in response["data"]]
            all_embeddings.extend(embeddings)

        # 转成 numpy
        return np.array(all_embeddings)
