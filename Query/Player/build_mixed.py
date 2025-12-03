"""
UDA-Bench Query Generation - MIXED Queries

This module generates Mixed queries that combine multiple operators:
- SELECT (required, 1 type)
- Filter/WHERE (6 subcategories)
- Aggregation/GROUP BY (1 type)
- Join (2 types: binary_join, multi_table_join)

Rules:
1. Every query MUST include SELECT
2. If Filter and Join appear together, attributes in WHERE must have table prefix
3. Each query must have at least 3 operators (SELECT + 2 others)

Possible combinations (with at least 3 operators):
- SELECT + Filter + Agg
- SELECT + Filter + Join  
- SELECT + Agg + Join
- SELECT + Filter + Agg + Join
"""

import os
import sys
import random
from typing import List, Dict, Tuple, Set
from itertools import product

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    Attribute, AttributeType, AttributeUsage, AttributeModality,
    DataStatistics, save_queries_to_file, load_attributes_from_json
)
from build_select import build_select_clause
from build_filter import build_where_clause, generate_predicate
from build_agg import build_aggregation_clause, build_group_by_clause
from build_join import (
    JoinGraph, JoinPath, TableConfig, 
    build_select_in_join_clause, build_from_join_clause,
    create_player_join_graph, create_med_join_graph, attrs_from_json
)


# =============================================================================
# Operator Definitions
# =============================================================================

# Operator types
OPERATORS = {
    "select": 1,      # 1 type
    "filter": 6,      # 6 subcategories
    "agg": 1,         # 1 type
    # "join": 2         # binary_join, multi_table_join
}

# Filter subcategory names
FILTER_SUBCATEGORIES = {
    1: "single",
    2: "two_and",
    3: "two_or",
    4: "multi_and",
    5: "multi_or",
    6: "mixed_and_or"
}

# Join subcategory names
JOIN_SUBCATEGORIES = {
    1: "binary_join",
    2: "multi_table_join"
}


# =============================================================================
# Generate Valid Operator Combinations
# =============================================================================

def get_valid_combinations() -> List[Tuple[str, ...]]:
    """
    Generate all valid operator combinations.
    
    Rules:
    - SELECT is always included (implicit)
    - At least 2 additional operators required
    
    Returns:
        List of operator tuples, e.g., [("filter", "agg"), ("filter", "join"), ...]
    """
    other_operators = ["filter", "agg", "join"]
    
    valid_combinations = []
    
    # 2 operators (SELECT + 2)
    for i, op1 in enumerate(other_operators):
        for op2 in other_operators[i+1:]:
            valid_combinations.append((op1, op2))
    
    # 3 operators (SELECT + 3)
    valid_combinations.append(("filter", "agg", "join"))
    
    return valid_combinations


def get_subcategory_combinations(operators: Tuple[str, ...]) -> List[Dict[str, int]]:
    """
    Generate all subcategory combinations for given operators.
    
    Args:
        operators: Tuple of operator names
    
    Returns:
        List of dicts mapping operator to subcategory number
    """
    subcategory_ranges = []
    operator_list = list(operators)
    
    for op in operator_list:
        if op == "filter":
            subcategory_ranges.append(range(1, 7))  # 1-6
        elif op == "join":
            subcategory_ranges.append(range(1, 3))  # 1-2
        else:
            subcategory_ranges.append(range(1, 2))  # just 1
    
    combinations = []
    for combo in product(*subcategory_ranges):
        combinations.append({op: subcat for op, subcat in zip(operator_list, combo)})
    
    return combinations


# =============================================================================
# WHERE Clause Builder with Table Prefix (for Join queries)
# =============================================================================

def build_where_clause_with_table_prefix(
    join_graph: JoinGraph,
    table_names: List[str],
    subcategory: int,
    filter_count: int = 3,
    seed: int = None
) -> Tuple[str, Dict]:
    """
    Build a WHERE clause with table-prefixed attributes for JOIN queries.
    
    Args:
        join_graph: JoinGraph with table configurations
        table_names: List of table names in the join
        subcategory: Filter subcategory (1-6)
        filter_count: Number of filters for subcategory 4, 5, 6
        seed: Random seed
    
    Returns:
        Tuple of (WHERE clause string, metadata dict)
    """
    if seed is not None:
        random.seed(seed)
    
    # Get all filterable attributes from all tables, only those with valid stats
    all_attrs = []
    for table in table_names:
        if table in join_graph.tables:
            table_attrs = join_graph.tables[table].attributes
            table_stats = join_graph.data_stats.get(table)
            for a in table_attrs:
                if a.modality != AttributeModality.IMAGE:
                    # Only include if has valid stats
                    if table_stats is not None and table_stats.get_column_info(a.name) is not None:
                        all_attrs.append(a)
    
    if not all_attrs:
        raise ValueError("No filterable attributes with valid stats available")
    
    # Initialize metadata
    metadata = {
        "filter_combination": "",
        "num_filters": 0,
        "predicates": [],
        "selectivity_ratios": []
    }
    
    def generate_predicate_with_prefix_retry(max_retries: int = 10) -> Tuple[str, float]:
        """Generate a valid predicate with table prefix, retrying if needed."""
        for _ in range(max_retries):
            attr = random.choice(all_attrs)
            selectivity = random.choice(["low", "medium", "high"])
            
            # Get stats for the specific table
            table_stats = join_graph.data_stats.get(attr.table)
            pred, ratio = generate_predicate(attr, selectivity, table_stats)
            
            if pred is not None:
                # Add table prefix
                prefixed_pred = pred.replace(attr.name, f"{attr.table}.{attr.name}", 1)
                return prefixed_pred, ratio
        
        raise ValueError("Could not generate valid predicate after max retries")
    
    def generate_n_predicates(n: int) -> Tuple[List[str], List[float]]:
        """Generate n predicates with retry."""
        predicates = []
        ratios = []
        for _ in range(n):
            pred, ratio = generate_predicate_with_prefix_retry()
            predicates.append(pred)
            ratios.append(ratio)
        return predicates, ratios
    
    if subcategory == 1:
        predicate, ratio = generate_predicate_with_prefix_retry()
        where_clause = predicate
        metadata["filter_combination"] = "single"
        metadata["num_filters"] = 1
        metadata["predicates"] = [predicate]
        metadata["selectivity_ratios"] = [ratio]
    
    elif subcategory == 2:
        predicates, ratios = generate_n_predicates(2)
        where_clause = " AND ".join(predicates)
        metadata["filter_combination"] = "AND"
        metadata["num_filters"] = 2
        metadata["predicates"] = predicates
        metadata["selectivity_ratios"] = ratios
    
    elif subcategory == 3:
        predicates, ratios = generate_n_predicates(2)
        where_clause = " OR ".join(predicates)
        metadata["filter_combination"] = "OR"
        metadata["num_filters"] = 2
        metadata["predicates"] = predicates
        metadata["selectivity_ratios"] = ratios
    
    elif subcategory == 4:
        n = max(3, filter_count)
        predicates, ratios = generate_n_predicates(n)
        where_clause = " AND ".join(predicates)
        metadata["filter_combination"] = "AND"
        metadata["num_filters"] = n
        metadata["predicates"] = predicates
        metadata["selectivity_ratios"] = ratios
    
    elif subcategory == 5:
        n = max(3, filter_count)
        predicates, ratios = generate_n_predicates(n)
        where_clause = " OR ".join(predicates)
        metadata["filter_combination"] = "OR"
        metadata["num_filters"] = n
        metadata["predicates"] = predicates
        metadata["selectivity_ratios"] = ratios
    
    elif subcategory == 6:
        n = max(4, filter_count)
        predicates, ratios = generate_n_predicates(n)
        mid = n // 2
        group1 = " AND ".join(predicates[:mid])
        group2 = " AND ".join(predicates[mid:])
        where_clause = f"({group1}) OR ({group2})"
        metadata["filter_combination"] = "AND_OR_mixed"
        metadata["num_filters"] = n
        metadata["predicates"] = predicates
        metadata["selectivity_ratios"] = ratios
    
    else:
        raise ValueError(f"Invalid subcategory: {subcategory}")
    
    return where_clause, metadata


# =============================================================================
# Aggregation Clause Builder with Table Prefix (for Join queries)
# =============================================================================

def build_aggregation_clause_with_prefix(
    join_graph: JoinGraph,
    table_names: List[str],
    num_agg_funcs: int = 1,
    seed: int = None
) -> Tuple[str, List[Dict]]:
    """
    Build aggregation expressions with table-prefixed attributes for JOIN queries.
    """
    if seed is not None:
        random.seed(seed)
    
    from build_agg import AGG_FUNCTIONS, NUMERIC_ONLY_AGG
    
    # Get all attributes from all tables
    all_attrs = []
    numerical_attrs = []
    non_image_attrs = []
    
    for table in table_names:
        if table in join_graph.tables:
            for attr in join_graph.tables[table].attributes:
                all_attrs.append(attr)
                if attr.usage == AttributeUsage.NUMERICAL:
                    numerical_attrs.append(attr)
                if attr.modality != AttributeModality.IMAGE:
                    non_image_attrs.append(attr)
    
    if not non_image_attrs:
        non_image_attrs = all_attrs
    
    agg_expressions = []
    agg_metadata = []
    
    for _ in range(num_agg_funcs):
        agg_func = random.choice(AGG_FUNCTIONS)
        
        if agg_func in NUMERIC_ONLY_AGG:
            if not numerical_attrs:
                agg_func = 'COUNT'
                agg_attr = random.choice(non_image_attrs)
            else:
                agg_attr = random.choice(numerical_attrs)
        else:
            agg_attr = random.choice(non_image_attrs)
        
        # Use table.attribute format
        full_name = f"{agg_attr.table}.{agg_attr.name}"
        alias = f"{agg_func.lower()}_{agg_attr.table}_{agg_attr.name}"
        agg_expr = f"{agg_func}({full_name}) AS {alias}"
        agg_expressions.append(agg_expr)
        
        agg_metadata.append({
            "function": agg_func,
            "attribute": full_name,
            "alias": alias
        })
    
    return ", ".join(agg_expressions), agg_metadata


def build_group_by_clause_with_prefix(
    join_graph: JoinGraph,
    table_names: List[str],
    num_group_by: int = 1,
    seed: int = None
) -> Tuple[str, List[Attribute]]:
    """
    Build GROUP BY clause with table-prefixed attributes for JOIN queries.
    """
    if seed is not None:
        random.seed(seed)
    
    # Get categorical attributes from all tables
    categorical_attrs = []
    for table in table_names:
        if table in join_graph.tables:
            for attr in join_graph.tables[table].attributes:
                if attr.usage == AttributeUsage.CATEGORICAL:
                    categorical_attrs.append(attr)
    
    if not categorical_attrs:
        raise ValueError("No categorical attributes available for GROUP BY")
    
    num_group_by = min(num_group_by, len(categorical_attrs))
    group_by_attrs = random.sample(categorical_attrs, num_group_by)
    
    # Build GROUP BY clause with table prefix
    group_by_names = ", ".join([f"{attr.table}.{attr.name}" for attr in group_by_attrs])
    group_by_clause = f"GROUP BY {group_by_names}"
    
    return group_by_clause, group_by_attrs


# =============================================================================
# Mixed Query Generator
# =============================================================================

def generate_mixed_query(
    operators: Tuple[str, ...],
    subcategories: Dict[str, int],
    # For non-join queries
    attributes: List[Attribute] = None,
    table: str = None,
    stats: DataStatistics = None,
    # For join queries
    join_graph: JoinGraph = None,
    table_names: List[str] = None,
    # Common params
    select_attr_num: int = 3,
    image_num: int = 0,
    filter_count: int = 3,
    num_group_by: int = 1,
    num_agg_funcs: int = 1,
    seed: int = None
) -> Tuple[str, Dict]:
    """
    Generate a mixed query combining multiple operators.
    
    Args:
        operators: Tuple of operator names (e.g., ("filter", "agg"))
        subcategories: Dict mapping operator to subcategory number
        attributes: List of Attribute objects (for non-join queries)
        table: Table name (for non-join queries)
        stats: DataStatistics object
        join_graph: JoinGraph (for join queries)
        table_names: List of table names (for join queries)
        select_attr_num: Number of SELECT attributes
        image_num: Number of image attributes
        filter_count: Number of filters
        num_group_by: Number of GROUP BY attributes
        num_agg_funcs: Number of aggregation functions
        seed: Random seed
    
    Returns:
        Tuple of (SQL query string, metadata dict)
    """
    if seed is not None:
        random.seed(seed)
    
    has_join = "join" in operators
    has_filter = "filter" in operators
    has_agg = "agg" in operators
    
    # Build subcategory name for display
    subcat_parts = []
    for op in operators:
        subcat_num = subcategories.get(op, 1)
        subcat_parts.append(f"{op}{subcat_num}")
    subcategory_name = "_".join(subcat_parts)
    
    metadata = {
        "category": "Mixed",
        "subcategory": subcategory_name,
        "operators": list(operators),
        "subcategories": subcategories,
    }
    
    # =========================================================================
    # Case 1: Has JOIN
    # =========================================================================
    if has_join:
        if join_graph is None or table_names is None:
            raise ValueError("join_graph and table_names required for join queries")
        
        join_subcat = subcategories.get("join", 1)
        
        # Determine tables based on join subcategory
        if join_subcat == 1:  # binary_join
            tables_to_use = table_names[:2]
        else:  # multi_table_join
            tables_to_use = table_names[:min(len(table_names), 4)]
            if len(tables_to_use) < 3:
                tables_to_use = table_names[:3] if len(table_names) >= 3 else table_names
        
        # Build FROM JOIN clause
        from_clause, join_conditions = build_from_join_clause(join_graph, tables_to_use)
        
        metadata["tables"] = tables_to_use
        metadata["join_conditions"] = join_conditions
        
        # Build WHERE clause if filter is included
        where_part = ""
        if has_filter:
            filter_subcat = subcategories.get("filter", 1)
            where_clause, where_meta = build_where_clause_with_table_prefix(
                join_graph, tables_to_use, filter_subcat, filter_count, seed
            )
            where_part = f" WHERE {where_clause}"
            metadata["filter_subcategory"] = filter_subcat
            metadata["filter_metadata"] = where_meta
        
        # Build SELECT and GROUP BY clauses based on whether agg is included
        if has_agg:
            # For aggregation queries: SELECT only GROUP BY attributes + aggregation functions
            # Build aggregation
            agg_clause, agg_meta = build_aggregation_clause_with_prefix(
                join_graph, tables_to_use, num_agg_funcs, seed
            )
            
            # Build GROUP BY
            group_by_clause, group_by_attrs = build_group_by_clause_with_prefix(
                join_graph, tables_to_use, num_group_by, seed
            )
            
            # Build SELECT: only GROUP BY attrs + aggregation functions
            group_by_select = ", ".join([f"{a.table}.{a.name}" for a in group_by_attrs])
            select_clause = f"SELECT {group_by_select}, {agg_clause}"
            group_by_part = f" {group_by_clause}"
            
            metadata["aggregations"] = agg_meta
            metadata["group_by_attributes"] = [f"{a.table}.{a.name}" for a in group_by_attrs]
            metadata["selected_attributes"] = [f"{a.table}.{a.name}" for a in group_by_attrs]
        else:
            # For non-aggregation queries: normal SELECT clause
            select_clause, selected_attrs = build_select_in_join_clause(
                join_graph, tables_to_use, select_attr_num, seed
            )
            group_by_part = ""
            metadata["selected_attributes"] = [f"{a.table}.{a.name}" for a in selected_attrs]
        
        # Combine all parts
        sql_query = f"{select_clause} {from_clause}{where_part}{group_by_part};"
    
    # =========================================================================
    # Case 2: No JOIN (simple table queries)
    # =========================================================================
    else:
        if attributes is None or table is None:
            raise ValueError("attributes and table required for non-join queries")
        
        # Determine the actual table first
        available_tables = list(set(attr.table for attr in attributes if attr.table))
        if not available_tables:
            actual_table = table
        elif table in available_tables:
            actual_table = table
        else:
            # Try partial match
            matching_tables = [t for t in available_tables if table in t or t in table]
            if matching_tables:
                actual_table = matching_tables[0]
            else:
                actual_table = available_tables[0]
        
        metadata["tables"] = [actual_table]
        table_attrs = [a for a in attributes if a.table == actual_table]
        
        # Build WHERE clause if filter is included
        where_part = ""
        if has_filter:
            filter_subcat = subcategories.get("filter", 1)
            where_clause, where_meta = build_where_clause(
                table_attrs, filter_subcat, filter_count, stats, seed
            )
            where_part = f" WHERE {where_clause}"
            metadata["filter_subcategory"] = filter_subcat
            metadata["filter_metadata"] = where_meta
        
        # Build SELECT and GROUP BY clauses based on whether agg is included
        if has_agg:
            # For aggregation queries: SELECT only GROUP BY attributes + aggregation functions
            # Build aggregation
            agg_clause, agg_meta = build_aggregation_clause(
                table_attrs, num_agg_funcs, seed
            )
            
            # Build GROUP BY
            group_by_clause, group_by_attrs = build_group_by_clause(
                table_attrs, num_group_by, seed
            )
            
            # Build SELECT: only GROUP BY attrs + aggregation functions
            group_by_select = ", ".join([a.name for a in group_by_attrs])
            select_clause = f"SELECT {group_by_select}, {agg_clause}"
            group_by_part = f" {group_by_clause}"
            
            metadata["aggregations"] = agg_meta
            metadata["group_by_attributes"] = [a.name for a in group_by_attrs]
            metadata["selected_attributes"] = [a.name for a in group_by_attrs]
        else:
            # For non-aggregation queries: normal SELECT clause
            select_clause, selected_attrs, _ = build_select_clause(
                table_attrs, select_attr_num, image_num, seed, table=actual_table
            )
            group_by_part = ""
            metadata["selected_attributes"] = [a.name for a in selected_attrs]
        
        # Combine all parts
        sql_query = f"{select_clause} FROM {actual_table}{where_part}{group_by_part};"
    
    return sql_query, metadata


# =============================================================================
# Batch Generation and Save
# =============================================================================

def generate_and_save_mixed_queries(
    attributes: List[Attribute],
    table: str,
    output_dir: str,
    num_queries_per_combination: int = 5,
    stats: DataStatistics = None,
    select_attr_num: int = 3,
    image_num: int = 0,
    join_graph: JoinGraph = None,
    join_table_names: List[str] = None,
    table_suffix: str = None
):
    """
    Generate all types of Mixed queries and save to files.
    
    Args:
        attributes: List of Attribute objects for single-table queries
        table: Table name for single-table queries
        output_dir: Directory to save query files
        num_queries_per_combination: Number of queries per subcategory combination
        stats: DataStatistics object
        select_attr_num: Number of SELECT attributes
        image_num: Number of image attributes
        join_graph: JoinGraph for join queries (optional, for multi-table datasets)
        join_table_names: List of table names for join queries (optional)
    
    Returns:
        List of all generated query dictionaries
    """
    all_queries = []
    
    # Check if JOIN is supported (multi-table dataset)
    has_join_support = join_graph is not None and join_table_names is not None
    
    # Get all valid operator combinations
    valid_combinations = get_valid_combinations()
    
    # Filter out JOIN combinations if no join_graph provided
    if not has_join_support:
        valid_combinations = [combo for combo in valid_combinations if "join" not in combo]
        print(f"  ⚠️ Single-table mode: JOIN combinations disabled")
    
    print(f"  Valid operator combinations: {len(valid_combinations)}")
    for combo in valid_combinations:
        print(f"    - SELECT + {' + '.join(combo)}")
    
    # Generate queries for each combination
    for operators in valid_combinations:
        subcategory_combos = get_subcategory_combinations(operators)
        print(f"\n  Generating: SELECT + {' + '.join(operators)}")
        print(f"    Subcategory combinations: {len(subcategory_combos)}")
        
        for subcat_combo in subcategory_combos:
            combo_name = "_".join([f"{op}{subcat_combo[op]}" for op in operators])
            
            for i in range(num_queries_per_combination):
                try:
                    sql, meta = generate_mixed_query(
                        operators=operators,
                        subcategories=subcat_combo,
                        attributes=attributes,
                        table=table,
                        stats=stats,
                        join_graph=join_graph,
                        table_names=join_table_names,
                        select_attr_num=select_attr_num,
                        image_num=image_num,
                        seed=hash(combo_name) + i * 100
                    )
                    
                    meta["combination_name"] = combo_name
                    all_queries.append({
                        "sql": sql,
                        "metadata": meta
                    })
                except Exception as e:
                    print(f"      ⚠️ Error for {combo_name}: {e}")
                    break
            
            # Print first example
            if all_queries:
                last_combo_queries = [q for q in all_queries if q["metadata"].get("combination_name") == combo_name]
                if last_combo_queries:
                    print(f"      [{combo_name}] Generated {len(last_combo_queries)} queries")
    
    # Save to files with table suffix
    if table_suffix:
        json_filename = f"mixed_queries_{table_suffix}.json"
        sql_filename = f"mixed_queries_{table_suffix}.sql"
    else:
        json_filename = "mixed_queries.json"
        sql_filename = "mixed_queries.sql"
    
    json_path = os.path.join(output_dir, json_filename)
    sql_path = os.path.join(output_dir, sql_filename)
    
    save_queries_to_file(all_queries, json_path, format="json")
    save_queries_to_file(all_queries, sql_path, format="sql")
    
    return all_queries


# =============================================================================
# Summary Statistics
# =============================================================================

def print_combination_summary():
    """Print summary of all possible combinations."""
    print("=" * 70)
    print("MIXED QUERY COMBINATIONS SUMMARY")
    print("=" * 70)
    
    valid_combinations = get_valid_combinations()
    
    total_subcategory_combos = 0
    
    for operators in valid_combinations:
        subcategory_combos = get_subcategory_combinations(operators)
        total_subcategory_combos += len(subcategory_combos)
        
        print(f"\nSELECT + {' + '.join(operators)}:")
        print(f"  Subcategory combinations: {len(subcategory_combos)}")
        
        # Print breakdown
        for op in operators:
            if op == "filter":
                print(f"    - filter: 6 subcategories")
            elif op == "join":
                print(f"    - join: 2 subcategories")
            else:
                print(f"    - {op}: 1 subcategory")
    
    print(f"\n{'─' * 70}")
    print(f"Total operator combinations: {len(valid_combinations)}")
    print(f"Total subcategory combinations: {total_subcategory_combos}")
    print("=" * 70)


# =============================================================================
# Main Entry Point
# =============================================================================

def generate_filter_agg_queries(base_path: str, output_path: str, num_queries_per_table: int = 6):
    """
    1. Generate FILTER+AGG queries (Single Table)
    All attributes in each SQL must come from the same table.
    """
    print("=" * 70)
    print("1. GENERATING FILTER+AGG QUERIES (Single Table)")
    print("=" * 70)
    
    attributes_path = f"{base_path}/Player_attributes.json"
    attributes = load_attributes_from_json(attributes_path)
    table_names = list(set(attr.table for attr in attributes if attr.table))
    
    all_queries = []
    for table_name in table_names:
        print(f"\n🔄 Processing {table_name} table...")
        
        # Filter attributes for this table only
        table_attrs = [attr for attr in attributes if attr.table == table_name]
        print(f"   Attributes: {len(table_attrs)}")
        
        # Load stats for this table
        table_csv_path = f"{base_path}/{table_name}.csv"
        if not os.path.exists(table_csv_path):
            print(f"   ⚠️ No CSV file found: {table_csv_path}")
            continue
            
        stats = DataStatistics(table_csv_path)
        print(f"   📊 Stats: {stats.total_rows} rows")
        
        try:
            # Generate filter+agg queries for this table
            queries = []
            for i in range(num_queries_per_table):
                sql, meta = generate_mixed_query(
                    operators=("filter", "agg"),
                    subcategories={"filter": (i % 6) + 1, "agg": 1},
                    attributes=table_attrs,
                    table=table_name,
                    stats=stats,
                    select_attr_num=3,
                    image_num=0,
                    seed=i * 100
                )
                meta["combination_name"] = f"filter{((i % 6) + 1)}_agg1"
                queries.append({"sql": sql, "metadata": meta})
            
            # Save queries for this table
            json_filename = f"mixed_queries_filter_agg_{table_name}.json"
            sql_filename = f"mixed_queries_filter_agg_{table_name}.sql"
            json_path = os.path.join(output_path, json_filename)
            sql_path = os.path.join(output_path, sql_filename)
            
            save_queries_to_file(queries, json_path, format="json")
            save_queries_to_file(queries, sql_path, format="sql")
            
            all_queries.extend(queries)
            print(f"   ✅ Generated {len(queries)} filter+agg queries for {table_name}")
            
        except Exception as e:
            print(f"   ❌ Error for {table_name}: {e}")
    
    print(f"\n📊 Total filter+agg queries: {len(all_queries)}")
    return all_queries


def generate_filter_join_queries(base_path: str, output_path: str, num_queries: int = 12):
    """
    2. Generate FILTER+JOIN queries (Multi-Table)
    """
    print("\n" + "=" * 70)
    print("2. GENERATING FILTER+JOIN QUERIES (Multi-Table)")
    print("=" * 70)
    
    # Create join graph
    join_graph = create_player_join_graph(base_path)
    join_table_names = ["player", "team", "city", "manager"]
    
    queries = []
    for i in range(num_queries):
        try:
            sql, meta = generate_mixed_query(
                operators=("filter", "join"),
                subcategories={"filter": (i % 6) + 1, "join": (i % 2) + 1},
                join_graph=join_graph,
                table_names=join_table_names,
                select_attr_num=4,
                image_num=0,
                seed=i * 200
            )
            meta["combination_name"] = f"filter{((i % 6) + 1)}_join{((i % 2) + 1)}"
            queries.append({"sql": sql, "metadata": meta})
            
        except Exception as e:
            print(f"   ⚠️ Error for query {i+1}: {e}")
    
    # Save filter+join queries
    json_path = os.path.join(output_path, "mixed_queries_filter_join.json")
    sql_path = os.path.join(output_path, "mixed_queries_filter_join.sql")
    
    save_queries_to_file(queries, json_path, format="json")
    save_queries_to_file(queries, sql_path, format="sql")
    
    print(f"✅ Generated {len(queries)} filter+join queries")
    return queries


def generate_agg_join_queries(base_path: str, output_path: str, num_queries: int = 4):
    """
    3. Generate AGG+JOIN queries (Multi-Table)
    """
    print("\n" + "=" * 70)
    print("3. GENERATING AGG+JOIN QUERIES (Multi-Table)")
    print("=" * 70)
    
    # Create join graph
    join_graph = create_player_join_graph(base_path)
    join_table_names = ["player", "team", "city", "manager"]
    
    queries = []
    for i in range(num_queries):
        try:
            sql, meta = generate_mixed_query(
                operators=("agg", "join"),
                subcategories={"agg": 1, "join": (i % 2) + 1},
                join_graph=join_graph,
                table_names=join_table_names,
                select_attr_num=4,
                image_num=0,
                seed=i * 250
            )
            meta["combination_name"] = f"agg1_join{((i % 2) + 1)}"
            queries.append({"sql": sql, "metadata": meta})
            
        except Exception as e:
            print(f"   ⚠️ Error for query {i+1}: {e}")
    
    # Save agg+join queries
    json_path = os.path.join(output_path, "mixed_queries_agg_join.json")
    sql_path = os.path.join(output_path, "mixed_queries_agg_join.sql")
    
    save_queries_to_file(queries, json_path, format="json")
    save_queries_to_file(queries, sql_path, format="sql")
    
    print(f"✅ Generated {len(queries)} agg+join queries")
    return queries


def generate_filter_agg_join_queries(base_path: str, output_path: str, num_queries: int = 12):
    """
    4. Generate FILTER+AGG+JOIN queries (Multi-Table)
    """
    print("\n" + "=" * 70)
    print("4. GENERATING FILTER+AGG+JOIN QUERIES (Multi-Table)")
    print("=" * 70)
    
    # Create join graph
    join_graph = create_player_join_graph(base_path)
    join_table_names = ["player", "team", "city", "manager"]
    
    queries = []
    for i in range(num_queries):
        try:
            sql, meta = generate_mixed_query(
                operators=("filter", "agg", "join"),
                subcategories={
                    "filter": (i % 6) + 1, 
                    "agg": 1,
                    "join": (i % 2) + 1
                },
                join_graph=join_graph,
                table_names=join_table_names,
                select_attr_num=4,
                image_num=0,
                seed=i * 300
            )
            meta["combination_name"] = f"filter{((i % 6) + 1)}_agg1_join{((i % 2) + 1)}"
            queries.append({"sql": sql, "metadata": meta})
            
        except Exception as e:
            print(f"   ⚠️ Error for query {i+1}: {e}")
    
    # Save filter+agg+join queries
    json_path = os.path.join(output_path, "mixed_queries_filter_agg_join.json")
    sql_path = os.path.join(output_path, "mixed_queries_filter_agg_join.sql")
    
    save_queries_to_file(queries, json_path, format="json")
    save_queries_to_file(queries, sql_path, format="sql")
    
    print(f"✅ Generated {len(queries)} filter+agg+join queries")
    return queries


if __name__ == "__main__":
    # Configuration
    base_path = "/data/dengqiyan/UDA-Bench/Query/Player"
    output_path = f"{base_path}/Mixed"
    os.makedirs(output_path, exist_ok=True)
    
    # Query generation parameters (you can adjust these numbers)
    FILTER_AGG_QUERIES_PER_TABLE = 6   # 6 queries per table (one for each filter subcategory)
    FILTER_JOIN_QUERIES = 12            # 12 filter+join queries  
    AGG_JOIN_QUERIES = 2                # 2 agg+join queries
    FILTER_AGG_JOIN_QUERIES = 12        # 12 filter+agg+join queries
    
    print("PLAYER DATASET MIXED QUERY GENERATION")
    print("=" * 70)
    print(f"📋 Configuration:")
    print(f"   Filter+Agg queries per table: {FILTER_AGG_QUERIES_PER_TABLE}")
    print(f"   Filter+Join queries: {FILTER_JOIN_QUERIES}")
    print(f"   Agg+Join queries: {AGG_JOIN_QUERIES}")
    print(f"   Filter+Agg+Join queries: {FILTER_AGG_JOIN_QUERIES}")
    
    # Generate queries in the specified order
    filter_agg_queries = generate_filter_agg_queries(base_path, output_path, FILTER_AGG_QUERIES_PER_TABLE)
    filter_join_queries = generate_filter_join_queries(base_path, output_path, FILTER_JOIN_QUERIES)
    agg_join_queries = generate_agg_join_queries(base_path, output_path, AGG_JOIN_QUERIES)
    filter_agg_join_queries = generate_filter_agg_join_queries(base_path, output_path, FILTER_AGG_JOIN_QUERIES)
    
    # Final summary
    total_queries = len(filter_agg_queries) + len(filter_join_queries) + len(agg_join_queries) + len(filter_agg_join_queries)
    print(f"\n📊 FINAL SUMMARY")
    print("=" * 70)
    print(f"1. Filter+Agg queries: {len(filter_agg_queries)}")
    print(f"2. Filter+Join queries: {len(filter_join_queries)}")
    print(f"3. Agg+Join queries: {len(agg_join_queries)}")
    print(f"4. Filter+Agg+Join queries: {len(filter_agg_join_queries)}")
    print(f"📊 Total queries: {total_queries}")
    print(f"📁 All queries saved to: {output_path}/")
    print("\nFiles generated:")
    print("   - mixed_queries_filter_agg_player.sql")
    print("   - mixed_queries_filter_agg_team.sql") 
    print("   - mixed_queries_filter_agg_city.sql")
    print("   - mixed_queries_filter_agg_manager.sql")
    print("   - mixed_queries_filter_join.sql")
    print("   - mixed_queries_agg_join.sql")
    print("   - mixed_queries_filter_agg_join.sql")
