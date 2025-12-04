"""
UDA-Bench Query Generation - AGGREGATION Queries

This module generates Aggregation queries following the template:
    SELECT {attribute(s)}, {agg_func}({attribute}) FROM {table} GROUP BY {categorical attribute(s)}

Rules:
- GROUP BY attributes must be categorical
- COUNT can be applied to any attribute
- SUM, AVG, MIN, MAX can only be applied to numerical attributes
"""

import os
import random
from typing import List, Dict, Tuple

from utils import Attribute, AttributeUsage, AttributeModality, save_queries_to_file
from build_select import build_select_clause


# =============================================================================
# Aggregation Function Definitions
# =============================================================================

AGG_FUNCTIONS = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX']
NUMERIC_ONLY_AGG = ['SUM', 'AVG', 'MIN', 'MAX']


# =============================================================================
# Aggregation Clause Builder
# =============================================================================

def build_aggregation_clause(
    attributes: List[Attribute],
    num_agg_funcs: int = 1,
    seed: int = None
) -> Tuple[str, List[Dict]]:
    """
    Build aggregation expressions for SELECT clause.
    
    Format: {agg_func}({attribute}) AS {alias}, ...
    
    Args:
        attributes: List of available Attribute objects
        num_agg_funcs: Number of aggregation functions to apply
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (aggregation clause string, list of aggregation metadata)
        e.g., ("COUNT(name) AS count_name, SUM(age) AS sum_age", [{...}, {...}])
    """
    if seed is not None:
        random.seed(seed)
    
    # Separate attributes by their properties
    numerical_attrs = [a for a in attributes if a.usage == AttributeUsage.NUMERICAL]
    non_image_attrs = [a for a in attributes if a.modality != AttributeModality.IMAGE]
    
    if not non_image_attrs:
        non_image_attrs = attributes
    
    agg_expressions = []
    agg_metadata = []
    
    for _ in range(num_agg_funcs):
        # Randomly choose aggregation function
        agg_func = random.choice(AGG_FUNCTIONS)
        
        if agg_func in NUMERIC_ONLY_AGG:
            # Must use numerical attribute
            if not numerical_attrs:
                agg_func = 'COUNT'  # Fallback to COUNT
                agg_attr = random.choice(non_image_attrs)
            else:
                agg_attr = random.choice(numerical_attrs)
        else:
            # COUNT can use any attribute (prefer non-image)
            agg_attr = random.choice(non_image_attrs)
        
        # Create aggregation expression with alias
        alias = f"{agg_func.lower()}_{agg_attr.name}"
        agg_expr = f"{agg_func}({agg_attr.name}) AS {alias}"
        agg_expressions.append(agg_expr)
        
        agg_metadata.append({
            "function": agg_func,
            "attribute": agg_attr.name,
            "alias": alias
        })
    
    agg_clause = ", ".join(agg_expressions)
    
    return agg_clause, agg_metadata


# =============================================================================
# GROUP BY Clause Builder
# =============================================================================

def build_group_by_clause(
    attributes: List[Attribute],
    num_group_by: int = 1,
    seed: int = None
) -> Tuple[str, List[Attribute]]:
    """
    Build a GROUP BY clause by selecting categorical attributes.
    
    Format: GROUP BY {attr1}, {attr2}, ...
    
    Args:
        attributes: List of available Attribute objects
        num_group_by: Number of GROUP BY attributes
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (GROUP BY clause string, list of selected attributes)
        e.g., ("GROUP BY position, nationality", [Attribute(...), ...])
    """
    if seed is not None:
        random.seed(seed)
    
    # Get categorical attributes for GROUP BY
    categorical_attrs = [a for a in attributes if a.usage == AttributeUsage.CATEGORICAL]
    
    if not categorical_attrs:
        raise ValueError("No categorical attributes available for GROUP BY")
    
    # Select GROUP BY attributes
    num_group_by = min(num_group_by, len(categorical_attrs))
    group_by_attrs = random.sample(categorical_attrs, num_group_by)
    
    # Build GROUP BY clause
    group_by_names = ", ".join([attr.name for attr in group_by_attrs])
    group_by_clause = f"GROUP BY {group_by_names}"
    
    return group_by_clause, group_by_attrs


# =============================================================================
# Full Aggregation Query Generator
# =============================================================================

def generate_agg_query(
    attributes: List[Attribute],
    table: str,
    num_group_by: int = 1,
    num_agg_funcs: int = 1,
    seed: int = None
) -> Tuple[str, Dict, str]:
    """
    Generate a complete Aggregation query.
    
    Template: SELECT {group_by_attrs}, {agg_clause} FROM {table} {group_by_clause}
    
    Args:
        attributes: List of available Attribute objects
        table: Table name hint (will match to actual table in attributes)
        num_group_by: Number of GROUP BY attributes
        num_agg_funcs: Number of aggregation functions to apply
        seed: Random seed
    
    Returns:
        Tuple of (SQL query string, metadata dict, actual table name)
    """
    if seed is not None:
        random.seed(seed)
    
    # First, determine the actual table from attributes
    available_tables = list(set(attr.table for attr in attributes if attr.table))
    
    if not available_tables:
        actual_table = table
        table_attrs = attributes
    elif table in available_tables:
        actual_table = table
        table_attrs = [a for a in attributes if a.table == actual_table]
    else:
        # Try partial match
        matching_tables = [t for t in available_tables if table in t or t in table]
        if matching_tables:
            actual_table = matching_tables[0]
        else:
            actual_table = available_tables[0]
        table_attrs = [a for a in attributes if a.table == actual_table]
    
    if not table_attrs:
        raise ValueError(f"No attributes found for table: {actual_table}")
    
    # Build GROUP BY clause first (to know which attrs are used)
    group_by_clause, group_by_attrs = build_group_by_clause(
        table_attrs, num_group_by, seed
    )
    
    # Build aggregation clause using only table_attrs
    agg_clause, agg_metadata = build_aggregation_clause(
        table_attrs, num_agg_funcs, seed
    )
    
    # Build SELECT clause: only GROUP BY attributes + aggregation functions
    group_by_select = ", ".join([attr.name for attr in group_by_attrs])
    final_select = f"SELECT {group_by_select}, {agg_clause}"
    
    # Build final SQL query
    sql_query = f"{final_select} FROM {actual_table} {group_by_clause};"
    
    metadata = {
        "category": "Agg",
        "subcategory": "aggregation",
        "tables": [actual_table],
        "group_by_attributes": [attr.name for attr in group_by_attrs],
        "aggregations": agg_metadata,
        "num_group_by": num_group_by,
        "num_aggregations": num_agg_funcs
    }
    
    return sql_query, metadata, actual_table


# =============================================================================
# Batch Generation and Save
# =============================================================================

def generate_and_save_agg_queries(
    attributes: List[Attribute],
    table: str,
    output_dir: str,
    num_queries: int = 10,
    num_group_by: int = 1,
    num_agg_funcs: int = 1
):
    """
    Generate multiple Aggregation queries and save to files.
    
    Args:
        attributes: List of available Attribute objects
        table: Table name
        output_dir: Directory to save the query files
        num_queries: Number of queries to generate
        num_group_by: Number of GROUP BY attributes
        num_agg_funcs: Number of aggregation functions to apply
    
    Returns:
        List of all generated query dictionaries
    """
    all_queries = []
    
    for i in range(num_queries):
        sql, meta, actual_table = generate_agg_query(
            attributes=attributes,
            table=table,
            num_group_by=num_group_by,
            num_agg_funcs=num_agg_funcs,
            seed=i * 42
        )
        all_queries.append({
            "sql": sql,
            "metadata": meta
        })
    
    file_json_name = f"agg_queries_{table}.json"
    file_sql_name = f"agg_queries_{table}.sql"
    json_path = os.path.join(output_dir, file_json_name)
    sql_path = os.path.join(output_dir, file_sql_name)
    # Save to JSON and SQL
    save_queries_to_file(all_queries, json_path, format="json")
    save_queries_to_file(all_queries, sql_path, format="sql")
    
    return all_queries


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    from utils import load_attributes_from_json
    
    # Configuration
    base_path = "/data2/jproject/OpenQuestProject/UDA-Bench/Query/Finan"
    attributes_path = f"{base_path}/Finan_attributes.json"
    output_path = f"{base_path}/Agg"
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Load attributes
    attributes = load_attributes_from_json(attributes_path)
    print(f"✅ Loaded {len(attributes)} attributes")
    
    # Test clause builders
    print("\n" + "=" * 70)
    print("Testing Clause Builders")
    print("=" * 70)
    
    # Test build_aggregation_clause
    agg_clause, agg_meta = build_aggregation_clause(attributes, num_agg_funcs=2)
    print(f"\nAggregation clause: {agg_clause}")
    print(f"Metadata: {agg_meta}")
    
    # Test build_group_by_clause
    group_by_clause, group_by_attrs = build_group_by_clause(attributes, num_group_by=2)
    print(f"\nGROUP BY clause: {group_by_clause}")
    print(f"Attributes: {[a.name for a in group_by_attrs]}")
    
    # Generate and save queries
    print("\n" + "=" * 70)
    print("GENERATING AGGREGATION QUERIES")
    print("=" * 70)
    
    queries = generate_and_save_agg_queries(
        attributes=attributes,
        table="Finan",
        output_dir=output_path,
        num_queries=8,
        num_group_by=1,
        num_agg_funcs=1
    )
    
    # Print examples
    print("\nGenerated queries:")
    print("-" * 70)
    for i, q in enumerate(queries[:5]):
        print(f"  {i+1}. {q['sql']}")
        print(f"      GROUP BY: {q['metadata']['group_by_attributes']}")
        print(f"      AGG: {q['metadata']['aggregations']}")
    if len(queries) > 5:
        print(f"  ... and {len(queries) - 5} more")
    
    print(f"\n📊 Summary:")
    print(f"   Total queries: {len(queries)}")
    print(f"   Saved to: {output_path}/")
