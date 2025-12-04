"""
匹配策略模块
定义了各种文本匹配策略的接口和实现
"""
from quest.conf.settings import LOCAL_MODEL_DIR
ROOT_MODEL_PATH = LOCAL_MODEL_DIR
THRESHOLD_JACCARD_EDIT_DISTANCE = 0.5

import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import re
from difflib import SequenceMatcher
import json
import requests
import time
import warnings

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    HAS_SEMANTIC_LIBS = True
except ImportError:
    HAS_SEMANTIC_LIBS = False
    print("Warning: sentence-transformers或sklearn未安装，语义相似度功能将不可用")


class MatchResult:
    """匹配结果类"""
    
    def __init__(self, is_match: bool, confidence: float, method: str, details: Dict[str, Any] = None):
        self.is_match = is_match
        self.confidence = confidence  # 0-1之间的置信度
        self.method = method
        self.details = details or {}
    
    def __str__(self):
        return f"MatchResult(match={self.is_match}, confidence={self.confidence:.4f}, method={self.method})"


class MatchStrategy(ABC):
    """匹配策略抽象基类"""
    
    def __init__(self, name: str, priority: int = 1, threshold: float = 0.8):
        self.name = name
        self.priority = priority  # 优先级，数字越小优先级越高
        self.threshold = threshold
    
    @abstractmethod
    def match(self, text1: str, text2: str) -> MatchResult:
        """执行匹配，返回匹配结果"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查策略是否可用"""
        pass
    
    def normalize_text(self, text: str) -> str:
        """标准化文本"""
        if not text:
            return ""
        text = str(text).strip().lower()
        text = re.sub(r'\s+', ' ', text)
        return text


class ExactMatchStrategy(MatchStrategy):
    """精确匹配策略"""
    
    def __init__(self, priority: int = 0):
        super().__init__("exact_match", priority, threshold=1.0)
    
    def match(self, text1: str, text2: str) -> MatchResult:
        norm1 = self.normalize_text(text1)
        norm2 = self.normalize_text(text2)
        
        if not norm1 and not norm2:
            return MatchResult(True, 1.0, self.name, {"normalized_text1": norm1, "normalized_text2": norm2})
        
        is_exact_match = norm1 == norm2
        confidence = 1.0 if is_exact_match else 0.0
        
        return MatchResult(is_exact_match, confidence, self.name, {
            "normalized_text1": norm1,
            "normalized_text2": norm2,
            "exact_match": is_exact_match
        })
    
    def is_available(self) -> bool:
        return True


def safe_split(cell, sep = '||'):
    # 防止空值
    if str(cell).strip() == "":
        return []
    return [s.strip() for s in str(cell).split(sep)]


def jaccard(a, b, threshold = THRESHOLD_JACCARD_EDIT_DISTANCE):
    """
    基于编辑距离的Jaccard相似度计算
    
    Args:
        a: 第一个字符串列表
        b: 第二个字符串列表
    
    Returns:
        float: Jaccard相似度 (交集大小 / 并集大小)
    """
    from difflib import SequenceMatcher
    
    # 标准化处理
    a = [str(i).strip().lower() for i in a if str(i).strip()]
    b = [str(j).strip().lower() for j in b if str(j).strip()]
    
    if not a and not b:
        return 1.0  # 两个空列表认为完全相似
    
    if not a or not b:
        return 0.0  # 一个为空一个不为空，相似度为0
    
    def edit_distance_similarity(s1, s2):
        """计算两个字符串的编辑距离相似度 (1 - 归一化编辑距离)"""
        return SequenceMatcher(None, s1, s2).ratio()
    
    # 找到实体匹配关系
    matched_from_a = set()  # 记录a中已匹配的索引
    matched_from_b = set()  # 记录b中已匹配的索引
    intersection_count = 0
    
    # 对a中的每个字符串，在b中找最佳匹配
    for i, str_a in enumerate(a):
        best_match_idx = -1
        best_similarity = 0
        
        for j, str_b in enumerate(b):
            if j in matched_from_b:  # 如果b中这个字符串已经被匹配过了，跳过
                continue
                
            similarity = edit_distance_similarity(str_a, str_b)
            if similarity >= threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match_idx = j
                jaccard_similarity = 1 # jznote 当前认为多值属性中只要有1个能匹配上就能匹配
                return jaccard_similarity
        
        # 如果找到了匹配
        if best_match_idx != -1:
            matched_from_a.add(i)
            matched_from_b.add(best_match_idx)
            intersection_count += 1
    
    # 计算并集大小：总的唯一实体数 = a的大小 + b的大小 - 交集大小
    union_count = len(a) + len(b) - intersection_count
    
    # 计算Jaccard相似度
    if union_count == 0:
        return 1.0
    
    jaccard_similarity = intersection_count / union_count
    if jaccard_similarity > 0.01:
        jaccard_similarity = 1
    return jaccard_similarity

class EditDistanceStrategy(MatchStrategy):
    """编辑距离匹配策略"""
    
    def __init__(self, threshold: float = 0.8, priority: int = 1):
        super().__init__("edit_distance", priority, threshold)
    
    def match(self, text1: str, text2: str) -> MatchResult:
        # 先检测是否包含 "||"， 如果是的话则记为多值属性。
        multiple_value_flag = False
        text1 = str(text1).strip()
        text2 = str(text2).strip()
        sep = "||"
        if "||" in text1 or "||" in text2:
            multiple_value_flag = True

        ###########
        norm1 = self.normalize_text(text1)
        norm2 = self.normalize_text(text2)
        
        if not norm1 and not norm2:
            return MatchResult(True, 1.0, self.name)
        
        if not norm1 or not norm2:
            return MatchResult(False, 0.0, self.name)
        
        if multiple_value_flag:
            a_split = safe_split(norm1, sep)

            b_split = safe_split(norm2, sep)
            threshold = self.threshold

            similarity = jaccard(a_split, b_split, threshold)
            is_match = similarity >= self.threshold

        else:
            similarity = SequenceMatcher(None, norm1, norm2).ratio()
            is_match = similarity >= self.threshold
        
        return MatchResult(is_match, similarity, self.name, {
            "similarity": similarity,
            "threshold": self.threshold,
            "normalized_text1": norm1,
            "normalized_text2": norm2
        })
    
    def is_available(self) -> bool:
        return True


class SemanticSimilarityStrategy(MatchStrategy):
    """语义相似度匹配策略"""
    
    def __init__(self, threshold: float = 0.8, priority: int = 2, model_name: str = "intfloat/multilingual-e5-large"):
        super().__init__("semantic_similarity", priority, threshold)
        self.model_name = model_name
        self.model_path = os.path.join(ROOT_MODEL_PATH, self.model_name)
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化语义模型"""
        if not HAS_SEMANTIC_LIBS:
            print(f"Warning: 无法加载语义模型 {self.model_name}，缺少必要的库")
            return
        
        try:
            print(f"正在加载语义模型: {self.model_name}")
            self.model = SentenceTransformer(self.model_path)
            print("语义模型加载成功")
        except Exception as e:
            print(f"Warning: 无法加载语义模型 {self.model_path}: {e}")
            self.model = None
    
    def match(self, text1: str, text2: str) -> MatchResult:
        if not self.is_available():
            return MatchResult(False, 0.0, self.name, {"error": "模型不可用"})
        
        norm1 = self.normalize_text(text1)
        norm2 = self.normalize_text(text2)
        
        if not norm1 and not norm2:
            return MatchResult(True, 1.0, self.name)
        
        if not norm1 or not norm2:
            return MatchResult(False, 0.0, self.name)
        
        try:
            # # 对于E5模型，需要添加指令前缀
            # if "e5" in self.model_name.lower():
            #     query_prefix = "query: "
            #     norm1 = query_prefix + norm1
            #     norm2 = query_prefix + norm2
            
            embeddings = self.model.encode([norm1, norm2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            similarity = float(similarity)
            
            is_match = similarity >= self.threshold
            
            return MatchResult(is_match, similarity, self.name, {
                "similarity": similarity,
                "threshold": self.threshold,
                "model": self.model_name,
                "normalized_text1": norm1,
                "normalized_text2": norm2
            })
            
        except Exception as e:
            print(f"语义相似度计算错误: {e}")
            return MatchResult(False, 0.0, self.name, {"error": str(e)})
    
    def is_available(self) -> bool:
        return self.model is not None and HAS_SEMANTIC_LIBS


class LLMJudgeStrategy(MatchStrategy):
    """大模型判断策略"""
    
    def __init__(self, threshold: float = 0.5, priority: int = 3, 
                 api_key: Optional[str] = None, api_base: Optional[str] = None,
                 model: str = "gpt-4.1-mini"):
        super().__init__("llm_judge", priority, threshold)
        self.api_key = api_key or "sk-DrStrjrlXohoCI2iBc14Eb41A38c428b88A0973aA29fCc3b"
        self.api_base = api_base or "https://aihubmix.com/v1"
        self.model = model
        self.max_retries = 3
        self.retry_delay = 1
    
    def _call_openai_api(self, text1: str, text2: str) -> Dict[str, Any]:
        """调用OpenAI API"""
        if not self.api_key:
            raise ValueError("需要设置API密钥才能使用LLM判断策略")
        
        prompt = f"""请判断以下两个文本是否表示相同的含义。
请只返回JSON格式的结果，包含以下字段：
- "is_match": true或false，表示是否匹配
- "confidence": 0-1之间的数字，表示判断的置信度
- "reasoning": 简短的判断理由

文本1: "{text1}"
文本2: "{text2}"

请直接返回JSON，不要包含其他内容："""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 200
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"].strip()
                    
                    # 尝试解析JSON
                    try:
                        parsed_result = json.loads(content)
                        return parsed_result
                    except json.JSONDecodeError:
                        # 如果不是纯JSON，尝试提取JSON部分
                        import re
                        json_match = re.search(r'\{.*\}', content, re.DOTALL)
                        if json_match:
                            parsed_result = json.loads(json_match.group())
                            return parsed_result
                        else:
                            raise ValueError(f"无法解析LLM响应: {content}")
                
                else:
                    print(f"API请求失败 (状态码: {response.status_code}): {response.text}")
                    
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
                print(f"API调用失败，正在重试 ({attempt + 1}/{self.max_retries}): {e}")
                time.sleep(self.retry_delay * (attempt + 1))
        
        raise Exception(f"API调用失败，已重试{self.max_retries}次")
    
    def match(self, text1: str, text2: str) -> MatchResult:
        if not self.is_available():
            return MatchResult(False, 0.0, self.name, {"error": "LLM API不可用"})
        
        norm1 = self.normalize_text(text1)
        norm2 = self.normalize_text(text2)
        
        if not norm1 and not norm2:
            return MatchResult(True, 1.0, self.name)
        
        if not norm1 or not norm2:
            return MatchResult(False, 0.0, self.name)
        
        try:
            llm_result = self._call_openai_api(text1, text2)  # 使用原始文本，不是标准化后的
            
            is_match = llm_result.get("is_match", False)
            confidence = float(llm_result.get("confidence", 0.0))
            reasoning = llm_result.get("reasoning", "")
            
            # 根据置信度和阈值判断最终结果
            final_match = is_match and confidence >= self.threshold
            
            return MatchResult(final_match, confidence, self.name, {
                "llm_is_match": is_match,
                "llm_confidence": confidence,
                "reasoning": reasoning,
                "threshold": self.threshold,
                "model": self.model,
                "original_text1": text1,
                "original_text2": text2
            })
            
        except Exception as e:
            print(f"LLM判断错误: {e}")
            return MatchResult(False, 0.0, self.name, {"error": str(e)})
    
    def is_available(self) -> bool:
        return self.api_key is not None


class MatchingEngine:
    """匹配引擎，负责管理和执行多种匹配策略"""
    
    def __init__(self, strategies: List[MatchStrategy], 
                 fusion_mode: str = "priority", 
                 early_stop_threshold: float = 0.95):
        """
        初始化匹配引擎
        
        Args:
            strategies: 匹配策略列表
            fusion_mode: 融合模式 ("priority", "voting", "weighted")
            early_stop_threshold: 早停阈值，当某个策略置信度超过此值时停止后续匹配
        """
        self.strategies = sorted(strategies, key=lambda x: x.priority)
        self.fusion_mode = fusion_mode
        self.early_stop_threshold = early_stop_threshold
        
        # 过滤不可用的策略
        self.available_strategies = [s for s in self.strategies if s.is_available()]
        
        if not self.available_strategies:
            raise ValueError("没有可用的匹配策略")
        
        print(f"匹配引擎初始化完成，可用策略: {[s.name for s in self.available_strategies]}")
    
    def match(self, text1: str, text2: str) -> MatchResult:
        """执行匹配"""
        results = []
        
        for strategy in self.available_strategies:
            #print(f"执行策略: {strategy.name}")
            result = strategy.match(text1, text2)
            results.append(result)
            
            #print(f"策略 {strategy.name} 结果: {result}")
            
            # 早停机制
            if result.confidence >= self.early_stop_threshold:
                #print(f"策略 {strategy.name} 置信度 {result.confidence:.4f} 超过早停阈值 {self.early_stop_threshold}，停止后续匹配")
                break
        
        # 根据融合模式决定最终结果
        if self.fusion_mode == "priority":
            return self._priority_fusion(results)
        elif self.fusion_mode == "voting":
            return self._voting_fusion(results)
        elif self.fusion_mode == "weighted":
            return self._weighted_fusion(results)
        else:
            raise ValueError(f"不支持的融合模式: {self.fusion_mode}")
    
    def _priority_fusion(self, results: List[MatchResult]) -> MatchResult:
        """优先级融合：返回第一个匹配的结果，如果都不匹配则返回最高优先级的结果"""
        # 寻找第一个匹配的结果
        for result in results:
            if result.is_match:
                return MatchResult(
                    True, 
                    result.confidence, 
                    f"priority_fusion({result.method})",
                    {"fusion_mode": "priority", "used_strategy": result.method, "all_results": [str(r) for r in results]}
                )
        
        # 如果都不匹配，返回最高优先级（第一个）的结果
        if results:
            best_result = results[0]
            return MatchResult(
                False,
                best_result.confidence,
                f"priority_fusion({best_result.method})",
                {"fusion_mode": "priority", "used_strategy": best_result.method, "all_results": [str(r) for r in results]}
            )
        
        return MatchResult(False, 0.0, "priority_fusion(no_results)")
    
    def _voting_fusion(self, results: List[MatchResult]) -> MatchResult:
        """投票融合：多数策略决定结果"""
        if not results:
            return MatchResult(False, 0.0, "voting_fusion(no_results)")
        
        match_votes = sum(1 for r in results if r.is_match)
        total_votes = len(results)
        
        is_final_match = match_votes > total_votes / 2
        avg_confidence = sum(r.confidence for r in results) / total_votes
        
        return MatchResult(
            is_final_match,
            avg_confidence,
            "voting_fusion",
            {
                "fusion_mode": "voting",
                "match_votes": match_votes,
                "total_votes": total_votes,
                "all_results": [str(r) for r in results]
            }
        )
    
    def _weighted_fusion(self, results: List[MatchResult]) -> MatchResult:
        """加权融合：根据策略优先级加权"""
        if not results:
            return MatchResult(False, 0.0, "weighted_fusion(no_results)")
        
        # 计算权重（优先级越高权重越大）
        max_priority = max(len(self.available_strategies) - i for i, _ in enumerate(self.available_strategies))
        
        weighted_score = 0
        total_weight = 0
        
        for i, result in enumerate(results):
            weight = max_priority - self.available_strategies[i].priority
            weighted_score += result.confidence * weight * (1 if result.is_match else 0)
            total_weight += weight
        
        final_confidence = weighted_score / total_weight if total_weight > 0 else 0
        is_final_match = final_confidence > 0.5
        
        return MatchResult(
            is_final_match,
            final_confidence,
            "weighted_fusion",
            {
                "fusion_mode": "weighted",
                "weighted_score": weighted_score,
                "total_weight": total_weight,
                "all_results": [str(r) for r in results]
            }
        )
