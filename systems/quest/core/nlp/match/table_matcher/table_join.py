"""
非精确表连接函数
实现不同表之间基于可能有差异的列（如大小写、缩写等）进行连接
"""

import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
import sys
from quest.conf.settings import JOIN_EDIT_DISTANCE_THRESHOLD, JOIN_SEMANTIC_THRESHOLD

import os

# 添加父目录到系统路径，以便导入ModularTableMatcher
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.append(parent_dir)

from quest.core.nlp.match.table_matcher.modular_matcher import ModularTableMatcher, MatchingConfig


def pd_join_by_column(left_table: pd.DataFrame, 
                       right_table: pd.DataFrame, 
                       left_column_name: str, 
                       right_column_name: str, 
                       column_type: str,
                       left_postfix: str = "_left", 
                       right_postfix: str = "_right",
                       matcher: Optional[ModularTableMatcher] = None) -> pd.DataFrame:
    """
    基于指定列进行非精确的表连接操作
    
    Args:
        left_table: 参与join的左侧表对应的pd.DataFrame
        right_table: 右侧表格的pd.DataFrame
        left_column_name: 左侧表格参与join的列
        right_column_name: 右侧表格参与join的列
        column_type: 参与join的列的类型，要求左右2表join的列的类型必须相同
        left_postfix: 如果left_column_name == right_column_name，join后的表格中为left_column_name加上的后缀名
        right_postfix: 如果left_column_name == right_column_name，join后的表格中为right_column_name加上的后缀名
        matcher: 可选的ModularTableMatcher实例，用于自定义匹配策略，如果为None则创建默认匹配器
        
    Returns:
        连接后的DataFrame
    """
    # 验证列是否存在于表中
    if left_column_name not in left_table.columns:
        raise ValueError(f"列 '{left_column_name}' 在左表中不存在")
    if right_column_name not in right_table.columns:
        raise ValueError(f"列 '{right_column_name}' 在右表中不存在")
    
    # 如果没有提供匹配器，则创建默认匹配器
    if matcher is None:
        matcher = ModularTableMatcher()
    
    # 准备结果表的行
    result_rows = []
    
    # 跟踪已匹配的右表行索引
    matched_right_indices = set()
    
    # 为右表创建索引到行的映射，以便快速访问
    right_idx_to_row = {idx: row for idx, row in right_table.iterrows()}
    
    # 处理列名冲突
    # 如果左右表有相同的列名（除了连接列外），需要对它们添加后缀
    left_columns = list(left_table.columns)
    right_columns = list(right_table.columns)
    
    # 处理重名列
    if left_column_name == right_column_name:
        # 连接列需要特殊处理
        join_column_left = f"{left_column_name}{left_postfix}"
        join_column_right = f"{left_column_name}{right_postfix}"
    else:
        join_column_left = left_column_name
        join_column_right = right_column_name
    
    # 为每一个左表的行寻找匹配的右表行
    print(f"开始匹配左表({len(left_table)}行)和右表({len(right_table)}行)...")
    
    for left_idx, left_row in left_table.iterrows():
        left_value = left_row[left_column_name]
        found_match = False
        
        for right_idx, right_row in right_table.iterrows():
            # 如果右表行已经匹配过，则跳过
            if right_idx in matched_right_indices:
                continue
            
            right_value = right_row[right_column_name]
            
            # 使用匹配器的match_values_by_type方法判断两个值是否匹配
            is_match, match_details = matcher.match_values_by_type(
                left_value, right_value, column_type
            )
            
            # 如果匹配成功
            if is_match:
                # 创建结果行
                result_row = {}
                
                # 处理左表的所有列
                for col in left_columns:
                    if col == left_column_name:
                        result_row[join_column_left] = left_row[col]
                    else:
                        # 处理左表的其他列
                        if col in right_columns and col != right_column_name:
                            # 如果列名在右表也存在，添加后缀
                            result_row[f"{col}{left_postfix}"] = left_row[col]
                        else:
                            # 如果列名不冲突，直接使用
                            result_row[col] = left_row[col]
                
                # 处理右表的所有列
                for col in right_columns:
                    if col == right_column_name:
                        result_row[join_column_right] = right_row[col]
                    else:
                        # 处理右表的其他列
                        if col in left_columns and col != left_column_name:
                            # 如果列名在左表也存在，添加后缀
                            result_row[f"{col}{right_postfix}"] = right_row[col]
                        else:
                            # 如果列名不冲突，直接使用
                            result_row[col] = right_row[col]
                
                # 添加匹配详情（可选）
                result_row['_match_confidence'] = match_details.get('confidence', 1.0)
                if 'match_result' in match_details:
                    result_row['_match_method'] = match_details['match_result'].method
                
                # 添加到结果行
                result_rows.append(result_row)
                
                # 标记右表行为已匹配
                matched_right_indices.add(right_idx)
                
                found_match = True
                break
        
        # 如果左表行没有找到匹配的右表行，可选是否添加（相当于左连接）
        # 这里实现的是内连接，所以不添加没有匹配的行
        if not found_match:
            pass
    
    # 创建结果DataFrame
    if result_rows:
        result_df = pd.DataFrame(result_rows)
    else:
        # 如果没有匹配的行，创建一个空的DataFrame，但保留列结构
        # 创建列名
        columns = []
        for col in left_columns:
            if col == left_column_name:
                columns.append(join_column_left)
            elif col in right_columns and col != right_column_name:
                columns.append(f"{col}{left_postfix}")
            else:
                columns.append(col)
                
        for col in right_columns:
            if col == right_column_name:
                columns.append(join_column_right)
            elif col in left_columns and col != left_column_name:
                columns.append(f"{col}{right_postfix}")
            elif col not in left_columns:
                columns.append(col)
                
        columns.extend(['_match_confidence', '_match_method'])
        result_df = pd.DataFrame(columns=columns)
    
    print(f"连接完成，匹配到 {len(result_df)} 行")
    return result_df



def create_advanced_group_by_matcher(semantic_similarity_threshold = 0.92 ) -> ModularTableMatcher:
    """创建针对表连接优化的匹配器"""
    config = (MatchingConfig()
             .add_exact_match(priority=0)
             .add_semantic_similarity(threshold = semantic_similarity_threshold, priority=2)
             .set_fusion_mode("priority"))
    return ModularTableMatcher(config)



def create_advanced_join_matcher(edit_distance_threshold = JOIN_EDIT_DISTANCE_THRESHOLD, semantic_similarity_threshold = JOIN_SEMANTIC_THRESHOLD ) -> ModularTableMatcher:
    """创建针对表连接优化的匹配器"""
    config = (MatchingConfig()
             .add_exact_match(priority=0)
             .add_edit_distance(threshold = edit_distance_threshold, priority=1)  # 较低的阈值以适应更多的变体
             .add_semantic_similarity(threshold = semantic_similarity_threshold, priority=2)
             .set_fusion_mode("priority"))
    return ModularTableMatcher(config)


# 支持不同类型的连接：内连接、左连接、右连接和全连接
def pd_join_by_column_with_join_type(left_table: pd.DataFrame, 
                                      right_table: pd.DataFrame, 
                                      left_column_name: str, 
                                      right_column_name: str, 
                                      column_type: str,
                                      join_type: str = 'inner',
                                      left_postfix: str = "_left", 
                                      right_postfix: str = "_right",
                                      matcher: Optional[ModularTableMatcher] = None) -> pd.DataFrame:
    """
    扩展的表连接函数，支持不同类型的连接操作（支持多对多匹配）
    
    Args:
        left_table: 参与join的左侧表对应的pd.DataFrame
        right_table: 右侧表格的pd.DataFrame
        left_column_name: 左侧表格参与join的列
        right_column_name: 右侧表格参与join的列
        column_type: 参与join的列的类型，要求左右2表join的列的类型必须相同
        join_type: 连接类型，可选 'inner'（内连接）、'left'（左连接）、'right'（右连接）、'outer'（全连接）
        left_postfix: 如果left_column_name == right_column_name，join后的表格中为left_column_name加上的后缀名
        right_postfix: 如果left_column_name == right_column_name，join后的表格中为right_column_name加上的后缀名
        matcher: 可选的ModularTableMatcher实例，用于自定义匹配策略，如果为None则创建默认匹配器
        
    Returns:
        连接后的DataFrame
    """
    # 验证连接类型
    valid_join_types = ['inner', 'left', 'right', 'outer']
    if join_type not in valid_join_types:
        raise ValueError(f"连接类型 '{join_type}' 不支持。支持的类型: {', '.join(valid_join_types)}")
    
    # 验证列是否存在于表中
    if left_column_name not in left_table.columns:
        raise ValueError(f"列 '{left_column_name}' 在左表中不存在")
    if right_column_name not in right_table.columns:
        raise ValueError(f"列 '{right_column_name}' 在右表中不存在")
    
    # 如果没有提供匹配器，则创建默认匹配器
    if matcher is None:
        matcher = ModularTableMatcher()
    
    # 准备结果表的行
    result_rows = []
    
    # 跟踪已匹配的行（用于处理左连接和全连接）
    matched_left_indices = set()
    matched_right_indices = set()  # 只用于跟踪右连接/全连接，不用于排除匹配
    
    # 列名处理
    if left_column_name == right_column_name:
        join_column_left = f"{left_column_name}{left_postfix}"
        join_column_right = f"{left_column_name}{right_postfix}"
    else:
        join_column_left = left_column_name
        join_column_right = right_column_name
    
    # 为每一个左表的行寻找匹配的右表行（支持多对多匹配）
    print(f"开始匹配左表({len(left_table)}行)和右表({len(right_table)}行)...")
    
    # 首先尝试匹配所有行
    for left_idx, left_row in left_table.iterrows():
        left_value = left_row[left_column_name]
        found_match = False
        
        # 对于每个左表行，遍历所有右表行进行匹配（不排除已匹配的右表行）
        for right_idx, right_row in right_table.iterrows():
            right_value = right_row[right_column_name]
            
            # 使用匹配器判断两个值是否匹配
            is_match, match_details = matcher.match_values_by_type(
                left_value, right_value, column_type
            )
            
            # 如果匹配成功
            if is_match:
                # 创建结果行，合并左右表数据
                result_row = _create_joined_row(
                    left_row, right_row,
                    left_table.columns, right_table.columns,
                    left_column_name, right_column_name,
                    join_column_left, join_column_right,
                    left_postfix, right_postfix,
                    match_details
                )
                
                result_rows.append(result_row)
                
                # 标记为已匹配（用于后续处理未匹配的行）
                matched_left_indices.add(left_idx)
                matched_right_indices.add(right_idx)
                
                found_match = True
                
                # 继续匹配其他可能的右表行（多对多匹配）
        
        # 处理左连接和全连接的情况：包括未匹配的左表行
        if not found_match and (join_type in ['left', 'outer']):
            result_row = _create_unmatched_left_row(
                left_row, 
                left_table.columns, right_table.columns,
                left_column_name, right_column_name,
                join_column_left, join_column_right,
                left_postfix, right_postfix
            )
            result_rows.append(result_row)
            matched_left_indices.add(left_idx)
    
    # 处理右连接和全连接的情况：包括未匹配的右表行
    if join_type in ['right', 'outer']:
        for right_idx, right_row in right_table.iterrows():
            if right_idx not in matched_right_indices:
                result_row = _create_unmatched_right_row(
                    right_row,
                    left_table.columns, right_table.columns,
                    left_column_name, right_column_name,
                    join_column_left, join_column_right,
                    left_postfix, right_postfix
                )
                result_rows.append(result_row)
    
    # 创建结果DataFrame
    if result_rows:
        result_df = pd.DataFrame(result_rows)
    else:
        # 创建空DataFrame但保留列结构
        columns = _get_result_columns(
            left_table.columns, right_table.columns,
            left_column_name, right_column_name,
            join_column_left, join_column_right,
            left_postfix, right_postfix
        )
        result_df = pd.DataFrame(columns=columns)
    
    print(f"连接完成，总共 {len(result_df)} 行")
    return result_df


def _create_joined_row(left_row, right_row, left_columns, right_columns,
                       left_column_name, right_column_name,
                       join_column_left, join_column_right,
                       left_postfix, right_postfix, match_details):
    """创建连接后的结果行"""
    result_row = {}
    
    # 处理左表的所有列
    for col in left_columns:
        if col == left_column_name:
            result_row[join_column_left] = left_row[col]
        else:
            # 处理左表的其他列
            if col in right_columns and col != right_column_name:
                # 如果列名在右表也存在，添加后缀
                result_row[f"{col}{left_postfix}"] = left_row[col]
            else:
                # 如果列名不冲突，直接使用
                result_row[col] = left_row[col]
    
    # 处理右表的所有列
    for col in right_columns:
        if col == right_column_name:
            result_row[join_column_right] = right_row[col]
        else:
            # 处理右表的其他列
            if col in left_columns and col != left_column_name:
                # 如果列名在左表也存在，添加后缀
                result_row[f"{col}{right_postfix}"] = right_row[col]
            else:
                # 如果列名不冲突，直接使用
                result_row[col] = right_row[col]
    
    # 添加匹配详情
    result_row['_match_confidence'] = match_details.get('confidence', 1.0)
    if 'match_result' in match_details:
        result_row['_match_method'] = match_details['match_result'].method
    
    return result_row


def _create_unmatched_left_row(left_row, left_columns, right_columns, 
                              left_column_name, right_column_name,
                              join_column_left, join_column_right,
                              left_postfix, right_postfix):
    """创建未匹配到右表的左表行"""
    result_row = {}
    
    # 处理左表的所有列
    for col in left_columns:
        if col == left_column_name:
            result_row[join_column_left] = left_row[col]
        else:
            # 处理左表的其他列
            if col in right_columns and col != right_column_name:
                # 如果列名在右表也存在，添加后缀
                result_row[f"{col}{left_postfix}"] = left_row[col]
            else:
                # 如果列名不冲突，直接使用
                result_row[col] = left_row[col]
    
    # 为右表的所有列添加空值
    for col in right_columns:
        if col == right_column_name:
            result_row[join_column_right] = None
        elif col in left_columns and col != left_column_name:
            result_row[f"{col}{right_postfix}"] = None
        else:
            result_row[col] = None
    
    # 添加匹配详情
    result_row['_match_confidence'] = 0.0
    result_row['_match_method'] = 'no_match'
    
    return result_row


def _create_unmatched_right_row(right_row, left_columns, right_columns, 
                               left_column_name, right_column_name,
                               join_column_left, join_column_right,
                               left_postfix, right_postfix):
    """创建未匹配到左表的右表行"""
    result_row = {}
    
    # 为左表的所有列添加空值
    for col in left_columns:
        if col == left_column_name:
            result_row[join_column_left] = None
        elif col in right_columns and col != right_column_name:
            result_row[f"{col}{left_postfix}"] = None
        else:
            result_row[col] = None
    
    # 处理右表的所有列
    for col in right_columns:
        if col == right_column_name:
            result_row[join_column_right] = right_row[col]
        else:
            # 处理右表的其他列
            if col in left_columns and col != left_column_name:
                # 如果列名在左表也存在，添加后缀
                result_row[f"{col}{right_postfix}"] = right_row[col]
            else:
                # 如果列名不冲突，直接使用
                result_row[col] = right_row[col]
    
    # 添加匹配详情
    result_row['_match_confidence'] = 0.0
    result_row['_match_method'] = 'no_match'
    
    return result_row


def _get_result_columns(left_columns, right_columns, left_column_name, right_column_name,
                       join_column_left, join_column_right, left_postfix, right_postfix):
    """获取结果DataFrame的列名"""
    columns = []
    
    # 处理左表列
    for col in left_columns:
        if col == left_column_name:
            columns.append(join_column_left)
        elif col in right_columns and col != right_column_name:
            columns.append(f"{col}{left_postfix}")
        else:
            columns.append(col)
    
    # 处理右表列
    for col in right_columns:
        if col == right_column_name:
            columns.append(join_column_right)
        elif col in left_columns and col != left_column_name:
            columns.append(f"{col}{right_postfix}")
        elif col not in left_columns:
            columns.append(col)
    
    # 添加匹配详情列
    columns.extend(['_match_confidence', '_match_method'])
    
    return columns
