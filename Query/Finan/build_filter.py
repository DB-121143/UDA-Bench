"""
UDA-Bench Query Generation - FILTER Queries

This module generates Filter queries with 6 subcategories:
    1: Single filter - WHERE {A}{O}{L}
    2: Two filters with AND - WHERE {A}{O}{L} AND {A}{O}{L}
    3: Two filters with OR - WHERE {A}{O}{L} OR {A}{O}{L}
    4: Multiple filters with AND - WHERE {A}{O}{L} (AND {A}{O}{L})^n
    5: Multiple filters with OR - WHERE {A}{O}{L} (OR {A}{O}{L})^n
    6: Mixed AND/OR combination - WHERE (... AND ...) OR (... AND ...)
"""

import random
from typing import List, Dict, Tuple

from utils import (
    Attribute, AttributeType, AttributeUsage, AttributeModality,
    DataStatistics
)
from build_select import build_select_clause


# =============================================================================
# Operator Definitions
# =============================================================================

NUMERIC_OPERATORS = ['=', '!=', '<', '<=', '>', '>=']
# STRING_OPERATORS = ['=', '!=', 'LIKE']
STRING_OPERATORS = ['=', '!=']
# CATEGORICAL_OPERATORS = ['=', '!=', 'IN']
CATEGORICAL_OPERATORS = ['=', '!=']


# =============================================================================
# Literal Generation
# =============================================================================

def has_valid_stats(attr: Attribute, stats: DataStatistics) -> bool:
    """
    Check if an attribute has valid statistics for literal generation.
    
    Args:
        attr: The attribute to check
        stats: DataStatistics object
    
    Returns:
        True if the attribute has valid stats, False otherwise
    """
    if stats is None:
        return False
    
    col_info = stats.get_column_info(attr.name)
    return col_info is not None


def generate_literal(
    attr: Attribute, 
    selectivity: str = "medium", 
    stats: DataStatistics = None
) -> Tuple[str, float]:
    """
    Generate a literal value for a given attribute based on real data statistics.
    
    Args:
        attr: The attribute to generate a literal for
        selectivity: 'low' (rare values, fewer results), 
                    'medium' (moderate frequency),
                    'high' (common values, more results)
        stats: DataStatistics object with real data
    
    Returns:
        Tuple of (literal_value_string, selectivity_ratio)
        Returns (None, 0.0) if no valid value can be generated
    """
    if stats is not None:
        # Try to get literal from real data
        literal, ratio = stats.get_literal_by_selectivity(attr.name, selectivity)
        
        if literal != "'unknown'":
            return literal, ratio
    
    # Return None to signal that caller should try another attribute
    return None, 0.0


def generate_predicate(
    attr: Attribute, 
    selectivity: str = "medium", 
    stats: DataStatistics = None
) -> Tuple[str, float]:
    """
    Generate a single filter predicate: {Attribute}{Operator}{Literal}
    
    Args:
        attr: The attribute for the predicate
        selectivity: Selectivity level for the literal
        stats: DataStatistics object for real data
    
    Returns:
        Tuple of (predicate_string, selectivity_ratio)
        Returns (None, 0.0) if no valid predicate can be generated
    """
    literal, ratio = generate_literal(attr, selectivity, stats)
    
    # If no valid literal, return None
    if literal is None:
        return None, 0.0
    
    # Choose operator based on attribute type
    if attr.value_type in [AttributeType.INTEGER, AttributeType.FLOAT]:
        operator = random.choice(NUMERIC_OPERATORS)
    elif attr.usage == AttributeUsage.CATEGORICAL:
        operator = random.choice(CATEGORICAL_OPERATORS)
    else:
        operator = random.choice(STRING_OPERATORS)
    
    return f"{attr.name} {operator} {literal}", ratio


def generate_predicate_with_retry(
    filterable_attrs: List[Attribute],
    stats: DataStatistics = None,
    max_retries: int = 10
) -> Tuple[Attribute, str, float]:
    """
    Generate a predicate, retrying with different attributes if needed.
    
    Args:
        filterable_attrs: List of attributes to choose from
        stats: DataStatistics object
        max_retries: Maximum number of retries
    
    Returns:
        Tuple of (selected_attribute, predicate_string, selectivity_ratio)
    """
    for _ in range(max_retries):
        attr = random.choice(filterable_attrs)
        selectivity = random.choice(["low", "medium", "high"])
        predicate, ratio = generate_predicate(attr, selectivity, stats)
        
        if predicate is not None:
            return attr, predicate, ratio
    
    # Fallback: should not reach here if stats are properly loaded
    raise ValueError("Could not generate valid predicate after max retries")


# =============================================================================
# WHERE Clause Builder
# =============================================================================

def build_where_clause(
    attributes: List[Attribute],
    subcategory: int,
    filter_count: int = 3,
    stats: DataStatistics = None,
    seed: int = None
) -> Tuple[str, Dict]:
    """
    Build a WHERE clause based on the subcategory.
    
    Subcategories:
        1: Single filter - WHERE {A}{O}{L}
        2: Two filters with AND - WHERE {A}{O}{L} AND {A}{O}{L}
        3: Two filters with OR - WHERE {A}{O}{L} OR {A}{O}{L}
        4: Multiple filters with AND - WHERE {A}{O}{L} (AND {A}{O}{L})^n
        5: Multiple filters with OR - WHERE {A}{O}{L} (OR {A}{O}{L})^n
        6: Mixed AND/OR combination - WHERE (... AND ...) OR (... AND ...)
    
    Args:
        attributes: List of available Attribute objects (should be filterable)
        subcategory: Filter subcategory (1-6)
        filter_count: Number of filters for subcategory 4, 5, 6 (default 3)
        stats: DataStatistics object for real data
        seed: Random seed
    
    Returns:
        Tuple of (WHERE clause string without "WHERE" prefix, metadata dict)
        e.g., ("age > 25 AND name = 'John'", {...})
    """
    if seed is not None:
        random.seed(seed)
    
    # Get filterable attributes (exclude image attributes for filtering)
    filterable_attrs = [a for a in attributes if a.modality != AttributeModality.IMAGE]
    
    # Filter to only attributes with valid stats
    if stats is not None:
        filterable_attrs = [a for a in filterable_attrs if has_valid_stats(a, stats)]
    
    if not filterable_attrs:
        raise ValueError("No filterable attributes with valid stats available")
    
    # Helper function to generate multiple predicates with retry
    def generate_n_predicates(n: int) -> Tuple[List[str], List[float]]:
        predicates = []
        ratios = []
        for _ in range(n):
            _, pred, ratio = generate_predicate_with_retry(filterable_attrs, stats)
            predicates.append(pred)
            ratios.append(ratio)
        return predicates, ratios
    
    # Initialize metadata
    metadata = {
        "filter_combination": "",
        "num_filters": 0,
        "predicates": [],
        "selectivity_ratios": []
    }
    
    if subcategory == 1:
        # Single filter
        _, predicate, ratio = generate_predicate_with_retry(filterable_attrs, stats)
        where_clause = predicate
        metadata["filter_combination"] = "single"
        metadata["num_filters"] = 1
        metadata["predicates"] = [predicate]
        metadata["selectivity_ratios"] = [ratio]
    
    elif subcategory == 2:
        # Two filters with AND
        predicates, ratios = generate_n_predicates(2)
        where_clause = " AND ".join(predicates)
        metadata["filter_combination"] = "AND"
        metadata["num_filters"] = 2
        metadata["predicates"] = predicates
        metadata["selectivity_ratios"] = ratios
    
    elif subcategory == 3:
        # Two filters with OR
        predicates, ratios = generate_n_predicates(2)
        where_clause = " OR ".join(predicates)
        metadata["filter_combination"] = "OR"
        metadata["num_filters"] = 2
        metadata["predicates"] = predicates
        metadata["selectivity_ratios"] = ratios
    
    elif subcategory == 4:
        # Multiple filters with AND (3 or more)
        n = max(3, filter_count)
        predicates, ratios = generate_n_predicates(n)
        where_clause = " AND ".join(predicates)
        metadata["filter_combination"] = "AND"
        metadata["num_filters"] = n
        metadata["predicates"] = predicates
        metadata["selectivity_ratios"] = ratios
    
    elif subcategory == 5:
        # Multiple filters with OR (3 or more)
        n = max(3, filter_count)
        predicates, ratios = generate_n_predicates(n)
        where_clause = " OR ".join(predicates)
        metadata["filter_combination"] = "OR"
        metadata["num_filters"] = n
        metadata["predicates"] = predicates
        metadata["selectivity_ratios"] = ratios
    
    elif subcategory == 6:
        # Mixed AND/OR combination: (p1 AND p2) OR (p3 AND p4) or similar CNF/DNF
        n = max(4, filter_count)
        predicates, ratios = generate_n_predicates(n)
        
        # Create groups for mixed combination
        mid = n // 2
        group1 = " AND ".join(predicates[:mid])
        group2 = " AND ".join(predicates[mid:])
        where_clause = f"({group1}) OR ({group2})"
        metadata["filter_combination"] = "AND_OR_mixed"
        metadata["num_filters"] = n
        metadata["predicates"] = predicates
        metadata["selectivity_ratios"] = ratios
    
    else:
        raise ValueError(f"Invalid subcategory: {subcategory}. Must be 1-6.")
    
    return where_clause, metadata


# =============================================================================
# Full Filter Query Generator
# =============================================================================

def generate_filter_query(
    attributes: List[Attribute],
    table: str,
    subcategory: int,
    select_attr_num: int = 2,
    image_num: int = 1,
    filter_count: int = 3,
    stats: DataStatistics = None,
    seed: int = None
) -> Tuple[str, Dict]:
    """
    Generate a complete Filter query based on the subcategory.
    
    Args:
        attributes: List of available Attribute objects
        table: Table name hint (will match to actual table in attributes)
        subcategory: Filter subcategory (1-6)
        select_attr_num: Number of attributes in SELECT clause
        image_num: Number of image attributes to include
        filter_count: Number of filters for subcategory 4, 5, 6 (default 3)
        stats: DataStatistics object for real data
        seed: Random seed
    
    Returns:
        Tuple of (SQL query string, metadata dict)
    """
    # Build SELECT clause - this also filters attributes to same table
    select_clause, selected_attrs, actual_table = build_select_clause(
        attributes, select_attr_num, image_num, seed, table=table
    )
    
    # Filter attributes to the same table for WHERE clause
    table_attrs = [a for a in attributes if a.table == actual_table]
    
    # Build WHERE clause using only attributes from the same table
    where_clause, where_metadata = build_where_clause(
        table_attrs, subcategory, filter_count, stats, seed
    )
    
    # Combine into full SQL query
    sql_query = f"{select_clause} FROM {actual_table} WHERE {where_clause};"
    
    # Build complete metadata
    metadata = {
        "category": "Filter",
        "subcategory": subcategory,
        "selected_attributes": [a.name for a in selected_attrs],
        "where_clause": where_clause,
        **where_metadata
    }
    
    return sql_query, metadata, actual_table


# =============================================================================
# Batch Generation and Save
# =============================================================================

SUBCATEGORY_NAMES = {
    1: "Single Filter",
    2: "Two Filters (AND)",
    3: "Two Filters (OR)",
    4: "Multiple Filters (AND)",
    5: "Multiple Filters (OR)",
    6: "Mixed AND/OR Combination"
}


def generate_and_save_filter_queries(
    attributes: List[Attribute],
    table: str,
    output_dir: str,
    num_queries_per_subcategory: int = 10,
    select_attr_num: int = 3,
    image_num: int = 0,
    filter_count: int = 4,
    stats: DataStatistics = None
):
    """
    Generate Filter queries for all 6 subcategories and save to files.
    
    Args:
        attributes: List of available Attribute objects
        table: Table name
        output_dir: Directory to save the query files
        num_queries_per_subcategory: Number of queries per subcategory
        select_attr_num: Number of attributes in SELECT clause
        image_num: Number of image attributes to include
        filter_count: Number of filters for subcategory 4, 5, 6
        stats: DataStatistics object for real data
    
    Returns:
        List of all generated query dictionaries
    """
    from utils import save_queries_to_file
    import os
    
    all_queries = []
    
    for subcategory in range(1, 7):
        print(f"  Generating subcategory {subcategory}: {SUBCATEGORY_NAMES[subcategory]}")
        
        for i in range(num_queries_per_subcategory):
            sql, meta, actual_table = generate_filter_query(
                attributes=attributes,
                table=table,
                subcategory=subcategory,
                select_attr_num=select_attr_num,
                image_num=image_num,
                filter_count=filter_count,
                stats=stats,
                seed=subcategory * 1000 + i * 42
            )
            
            # Add tables info for save_queries_to_file
            meta["tables"] = [actual_table]
            
            all_queries.append({
                "sql": sql,
                "metadata": meta
            })
    
    # Save to JSON and SQL
    file_json_name = f"filter_queries_{table}.json"
    file_sql_name = f"filter_queries_{table}.sql"
    json_path = os.path.join(output_dir, file_json_name)
    sql_path = os.path.join(output_dir, file_sql_name)
    
    save_queries_to_file(all_queries, json_path, format="json")
    save_queries_to_file(all_queries, sql_path, format="sql")
    
    return all_queries


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    from utils import load_attributes_from_json, DataStatistics
    
    # Configuration
    base_path = "/data/dengqiyan/UDA-Bench/Query/Player"
    attributes_path = f"{base_path}/Player_attributes.json"
    gt_data_path = f"{base_path}/Player.csv"

    output_path = f"{base_path}/Filter"
    
    # Load attributes and statistics
    attributes = load_attributes_from_json(attributes_path)
    data_stats = DataStatistics(gt_data_path)
    
    print(f"✅ Loaded {len(attributes)} attributes")
    print(f"📊 Analyzed {len(data_stats.column_stats)} columns from {data_stats.total_rows} rows")
    
    # Generate and save Filter queries
    print("\n" + "=" * 70)
    print("GENERATING FILTER QUERIES (6 Subcategories)")
    print("=" * 70)
    
    queries = generate_and_save_filter_queries(
        attributes=attributes,
        table="Player",
        output_dir=output_path,
        num_queries_per_subcategory=10,
        select_attr_num=3,
        image_num=1,
        filter_count=4,
        stats=data_stats
    )
    
    # Print summary
    print("\n" + "-" * 70)
    print("Generated queries by subcategory:")
    for subcat in range(1, 7):
        count = sum(1 for q in queries if q['metadata']['subcategory'] == subcat)
        print(f"  {subcat}. {SUBCATEGORY_NAMES[subcat]}: {count} queries")
    
    # Print examples
    print("\nExample queries:")
    print("-" * 70)
    for subcat in range(1, 7):
        example = next(q for q in queries if q['metadata']['subcategory'] == subcat)
        print(f"  [{subcat}] {example['sql'][:80]}...")
    
    print(f"\n📊 Summary:")
    print(f"   Total queries: {len(queries)}")
    print(f"   Saved to: {base_path}/")
