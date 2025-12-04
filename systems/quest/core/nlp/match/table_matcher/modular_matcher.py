"""
模块化表格匹配器
支持可插拔的匹配策略和灵活的配置
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Tuple, Optional
import re
import warnings
from quest.core.nlp.match.table_matcher.match_strategies import (
    MatchStrategy, MatchingEngine, MatchResult,
    ExactMatchStrategy, EditDistanceStrategy, 
    SemanticSimilarityStrategy, LLMJudgeStrategy
)

warnings.filterwarnings('ignore')


class MatchingConfig:
    """匹配配置类"""
    
    def __init__(self):
        self.strategies = []
        self.fusion_mode = "priority"
        self.early_stop_threshold = 0.99
        self.openai_api_key = None
        self.openai_api_base = None
        
    def add_exact_match(self, priority: int = 0) -> 'MatchingConfig':
        """添加精确匹配策略"""
        self.strategies.append(ExactMatchStrategy(priority=priority))
        return self
    
    def add_edit_distance(self, threshold: float = 0.8, priority: int = 1) -> 'MatchingConfig':
        """添加编辑距离匹配策略"""
        self.strategies.append(EditDistanceStrategy(threshold=threshold, priority=priority))
        return self
    
    def add_semantic_similarity(self, threshold: float = 0.8, priority: int = 2, 
                               model_name: str = "intfloat/multilingual-e5-large") -> 'MatchingConfig':
        """添加语义相似度匹配策略"""
        self.strategies.append(SemanticSimilarityStrategy(
            threshold=threshold, priority=priority, model_name=model_name))
        return self
    
    def add_llm_judge(self, threshold: float = 0.5, priority: int = 3,
                     model: str = "gpt-4o-mini") -> 'MatchingConfig':
        """添加LLM判断策略"""
        self.strategies.append(LLMJudgeStrategy(
            threshold=threshold, priority=priority,
            api_key=self.openai_api_key, api_base=self.openai_api_base,
            model=model))
        return self
    
    def set_fusion_mode(self, mode: str) -> 'MatchingConfig':
        """设置融合模式"""
        if mode not in ["priority", "voting", "weighted"]:
            raise ValueError(f"不支持的融合模式: {mode}")
        self.fusion_mode = mode
        return self
    
    def set_early_stop_threshold(self, threshold: float) -> 'MatchingConfig':
        """设置早停阈值"""
        self.early_stop_threshold = threshold
        return self
    
    def set_openai_config(self, api_key: str, api_base: Optional[str] = None) -> 'MatchingConfig':
        """设置OpenAI配置"""
        self.openai_api_key = api_key
        self.openai_api_base = api_base
        # 更新已有的LLM策略的API配置
        for strategy in self.strategies:
            if isinstance(strategy, LLMJudgeStrategy):
                strategy.api_key = api_key
                if api_base:
                    strategy.api_base = api_base
        return self
    
    @classmethod
    def default_config(cls) -> 'MatchingConfig':
        """获取默认配置"""
        return (cls()
                .add_exact_match(priority=0)
                .add_semantic_similarity(threshold=0.8, priority=1)
                .set_fusion_mode("priority")
                .set_early_stop_threshold(1.0))  # 精确匹配时早停
    
    @classmethod
    def comprehensive_config(cls, openai_api_key: Optional[str] = None) -> 'MatchingConfig':
        """获取全面的配置（包含所有策略）"""
        config = (cls()
                 .add_exact_match(priority=0)
                 .add_edit_distance(threshold=0.8, priority=1)
                 .add_semantic_similarity(threshold=0.75, priority=2)
                 .set_fusion_mode("priority")
                 .set_early_stop_threshold(0.95))
        
        if openai_api_key:
            config.set_openai_config(openai_api_key)
            config.add_llm_judge(threshold=0.5, priority=3)
        
        return config

    @classmethod
    def llm_judge_config(cls, openai_api_key: Optional[str] = None) -> 'MatchingConfig':
        """获取全面的配置（包含所有策略）"""
        config = (cls()
                 .add_exact_match(priority=0)
                 .add_llm_judge(threshold=0.5, priority=1)
                 .set_fusion_mode("priority"))
        
        return config

class ModularTableMatcher:
    """模块化表格匹配器"""
    
    def __init__(self, config: Optional[MatchingConfig] = None):
        """
        初始化模块化表格匹配器
        
        Args:
            config: 匹配配置，如果为None则使用默认配置
        """
        if config is None:
            config = MatchingConfig.default_config()
        
        self.config = config
        
        # 创建匹配引擎
        if not config.strategies:
            raise ValueError("至少需要配置一个匹配策略")
        
        try:
            self.matching_engine = MatchingEngine(
                config.strategies,
                config.fusion_mode,
                config.early_stop_threshold
            )
        except Exception as e:
            print(f"匹配引擎初始化失败: {e}")
            # 如果初始化失败，尝试使用最基本的策略
            fallback_strategies = [ExactMatchStrategy(), EditDistanceStrategy()]
            self.matching_engine = MatchingEngine(fallback_strategies, "priority", 1.0)
            print("使用基础策略作为备选方案")
    
    def normalize_int_value(self, value) -> Union[int, None]:
        """标准化整数值"""
        if pd.isna(value) or value is None or value == '':
            return None
        
        str_value = str(value).strip()
        str_value = re.sub(r'[,\s]', '', str_value)
        
        try:
            return int(float(str_value))
        except (ValueError, TypeError):
            return None
    
    def normalize_float_value(self, value) -> Union[float, None]:
        """标准化浮点数值"""
        if pd.isna(value) or value is None or value == '':
            return None
        
        str_value = str(value).strip()
        str_value = re.sub(r'[,\s]', '', str_value)
        
        try:
            return float(str_value)
        except (ValueError, TypeError):
            return None
    
    def match_text_values(self, ground_truth: str, extracted: str) -> Tuple[bool, Dict[str, any]]:
        """使用配置的策略匹配文本值"""
        result = self.matching_engine.match(ground_truth, extracted)
        return result.is_match, {
            "match_result": result,
            "confidence": result.confidence,
            "method": result.method,
            "details": result.details
        }
    
    def match_values_by_type(self, ground_truth, extracted, data_type: str = 'STRING') -> Tuple[bool, Dict[str, any]]:
        """根据数据类型匹配值"""
        data_type = data_type.upper()
        
        if data_type in ['INT', 'INTEGER', 'BIGINT', 'SMALLINT']:
            gt_val = self.normalize_int_value(ground_truth)
            ext_val = self.normalize_int_value(extracted)
            is_match = (gt_val == ext_val)
            return is_match, {
                "ground_truth_normalized": gt_val,
                "extracted_normalized": ext_val,
                "match_type": "exact_integer"
            }
        
        elif data_type in ['FLOAT', 'DOUBLE', 'DECIMAL', 'NUMERIC', 'REAL']:
            gt_val = self.normalize_float_value(ground_truth)
            ext_val = self.normalize_float_value(extracted)
            if gt_val is None and ext_val is None:
                return True, {"match_type": "both_null"}
            if gt_val is None or ext_val is None:
                return False, {"match_type": "one_null", "gt_val": gt_val, "ext_val": ext_val}
            is_match = abs(gt_val - ext_val) < 1e-6
            return is_match, {
                "ground_truth_normalized": gt_val,
                "extracted_normalized": ext_val,
                "difference": abs(gt_val - ext_val) if gt_val is not None and ext_val is not None else None,
                "match_type": "float_comparison"
            }
        
        elif data_type in ['VARCHAR', 'TEXT', 'STRING', 'CHAR']:
            is_match, details = self.match_text_values(ground_truth, extracted)
            details["match_type"] = "text_matching"
            return is_match, details
        
        else:
            # 默认使用文本匹配
            is_match, details = self.match_text_values(ground_truth, extracted)
            details["match_type"] = "default_text_matching"
            return is_match, details
    
    def create_composite_key(self, row: pd.Series, key_columns: List[str]) -> str:
        """创建复合主键"""
        key_parts = []
        for col in key_columns:
            value = str(row[col]) if pd.notna(row[col]) else ""
            key_parts.append(value)
        return "|".join(key_parts)
    
    def match_tables(self, 
                    ground_truth: pd.DataFrame, 
                    llm_extract: pd.DataFrame,
                    primary_keys: Union[str, List[str]],
                    column_types: Dict[str, str]) -> Dict[str, any]:
        """
        匹配两个表格并计算准确率
        
        Args:
            ground_truth: 真实数据表
            llm_extract: LLM抽取的数据表
            primary_keys: 主键列名（单个字符串或列表）
            column_types: 非主键列的类型字典
        
        Returns:
            包含匹配结果和准确率的字典
        """
        # 标准化主键参数
        if isinstance(primary_keys, str):
            key_columns = [primary_keys]
        else:
            key_columns = primary_keys
        
        # 验证列是否存在
        for col in key_columns:
            if col not in ground_truth.columns:
                raise ValueError(f"主键列 '{col}' 在ground_truth表中不存在")
            if col not in llm_extract.columns:
                raise ValueError(f"主键列 '{col}' 在llm_extract表中不存在")
        
        # 创建主键映射
        gt_with_key = ground_truth.copy()
        ext_with_key = llm_extract.copy()
        
        gt_with_key['__composite_key__'] = gt_with_key.apply(
            lambda row: self.create_composite_key(row, key_columns), axis=1
        )
        ext_with_key['__composite_key__'] = ext_with_key.apply(
            lambda row: self.create_composite_key(row, key_columns), axis=1
        )
        
        # 建立键到索引的映射
        gt_key_to_idx = gt_with_key.set_index('__composite_key__')
        ext_key_to_idx = ext_with_key.set_index('__composite_key__')
        
        # 获取非主键列
        non_key_columns = [col for col in ground_truth.columns if col not in key_columns]
        
        # 初始化结果
        results = {
            'column_precision': {},
            'column_recall': {},
            'column_f1_score': {},
            'overall_recall': 0.0,
            'overall_f1_score': 0.0,
            'overall_precision': 0.0,
            'total_rows': 0,
            'matched_rows': 0,
            'detailed_results': {},
            'matching_config': {
                'strategies': [s.name for s in self.config.strategies],
                'fusion_mode': self.config.fusion_mode,
                'early_stop_threshold': self.config.early_stop_threshold
            }
        }
        
        matched_keys = set(gt_key_to_idx.index) & set(ext_key_to_idx.index)
        results['total_rows'] = len(matched_keys)
        results['matched_rows'] = len(matched_keys)
        
        if not matched_keys:
            print("Warning: 没有找到匹配的主键")
            return results
        
        print(f"开始匹配 {len(matched_keys)} 行数据...")
        
        # 逐列计算准确率
        for column in non_key_columns:
            if column not in llm_extract.columns:
                print(f"Warning: 列 '{column}' 在llm_extract表中不存在，跳过")
                continue
            
            print(f"正在匹配列: {column}")
            
            column_type = column_types.get(column, 'TEXT')
            correct_count = 0
            num_matched_extracted_rows = len(matched_keys)
            num_T_Q = len(set(ext_key_to_idx.index))
            num_GT_Q = len(set(gt_key_to_idx.index))
            
            column_details = []
            
            for key in matched_keys:
                gt_value = gt_key_to_idx.loc[key, column]
                ext_value = ext_key_to_idx.loc[key, column]
                
                is_match, match_details = self.match_values_by_type(gt_value, ext_value, column_type)
                if is_match:
                    correct_count += 1
                
                column_details.append({
                    'key': key,
                    'ground_truth': gt_value,
                    'extracted': ext_value,
                    'match': is_match,
                    'type': column_type,
                    'match_details': match_details
                })
            
            precision = correct_count / num_T_Q if num_GT_Q  > 0 else 0.0
            recall = correct_count / num_GT_Q if num_GT_Q > 0 else 0.0
            f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            results['column_precision'][column] = precision
            results['column_recall'][column] = recall
            results['column_f1_score'][column] = f1_score
            results['detailed_results'][column] = column_details
            
            print(f"列 {column} precision: {precision:.4f}")
        
        # 计算总体precision、 recall和F1分数
        if results['column_precision']:
            results['overall_precision'] = np.mean(list(results['column_precision'].values()))
        if results['column_recall']:
            results['overall_recall'] = np.mean(list(results['column_recall'].values()))
        if results['column_f1_score']:
            results['overall_f1_score'] = np.mean(list(results['column_f1_score'].values()))

        
        return results
    
    def print_results(self, results: Dict[str, any], show_details: bool = False):
        """打印匹配结果"""
        print("=" * 80)
        print("模块化表格匹配结果报告")
        print("=" * 80)
        print(f"总行数: {results['total_rows']}")
        print(f"匹配行数: {results['matched_rows']}")
        print(f"总体precision: {results['overall_precision']:.4f}")
        print(f"总体recall: {results['overall_recall']:.4f}")
        print(f"总体F1分数: {results['overall_f1_score']:.4f}")
        print("-" * 80)

        print("\n匹配配置:")
        config = results.get('matching_config', {})
        print(f"  策略: {', '.join(config.get('strategies', []))}")
        print(f"  融合模式: {config.get('fusion_mode', 'unknown')}")
        print(f"  早停阈值: {config.get('early_stop_threshold', 'unknown')}")

        print("\n各列precision, recall, F1分数:")
        print("-" * 60)
        for column, precision in results['column_precision'].items():
            precision = results['column_precision'].get(column, 0.0)
            recall = results['column_recall'].get(column, 0.0)
            f1_score = results['column_f1_score'].get(column, 0.0)
            print(f"{column:30}: {precision:.4f}, {recall:.4f}, {f1_score:.4f}")
        print("-" * 60)
        show_details = False
        if show_details:
            print("\n详细匹配结果（显示前3个不匹配的案例）:")
            print("-" * 80)
            
            for column, details in results['detailed_results'].items():
                print(f"\n列: {column}")
                mismatches = [d for d in details if not d['match']]
                if mismatches:
                    print(f"  不匹配案例 (显示前3个):")
                    for i, detail in enumerate(mismatches[:3]):
                        print(f"    {i+1}. Key: {detail['key']}")
                        print(f"       Ground Truth: {detail['ground_truth']}")
                        print(f"       Extracted: {detail['extracted']}")
                        
                        # 显示匹配细节
                        match_details = detail.get('match_details', {})
                        if 'match_result' in match_details:
                            match_result = match_details['match_result']
                            print(f"       方法: {match_result.method}")
                            print(f"       置信度: {match_result.confidence:.4f}")
                            if hasattr(match_result, 'details') and match_result.details:
                                reasoning = match_result.details.get('reasoning', '')
                                if reasoning:
                                    print(f"       原因: {reasoning}")
                else:
                    print("  所有值都匹配!")


# 便捷的工厂函数
def create_simple_matcher() -> ModularTableMatcher:
    """创建简单的匹配器（只使用编辑距离）"""
    config = MatchingConfig().add_edit_distance(threshold=0.8)
    return ModularTableMatcher(config)


def create_semantic_matcher() -> ModularTableMatcher:
    """创建语义匹配器（精确匹配 + 语义相似度）"""
    config = MatchingConfig.default_config()
    return ModularTableMatcher(config)


def create_comprehensive_matcher(openai_api_key: Optional[str] = None) -> ModularTableMatcher:
    """创建全面的匹配器（包含所有策略）"""
    config = MatchingConfig.comprehensive_config(openai_api_key)
    return ModularTableMatcher(config)


def create_llm_judge_matcher(openai_api_key: Optional[str] = None) -> ModularTableMatcher:
    """创建LLM判断匹配器（只使用LLM判断）"""
    config = MatchingConfig.llm_judge_config(openai_api_key)
    return ModularTableMatcher(config)

