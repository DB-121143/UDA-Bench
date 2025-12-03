"""
UDA-Bench Query Generation - JOIN Queries

This module generates JOIN queries with two subcategories:
1. Binary Joins: SELECT {attrs} FROM {t1} JOIN {t2} ON {t1.key} = {t2.key}
2. Multi-table Joins: SELECT {attrs} FROM {t1} JOIN {t2} ON ... JOIN {t3} ON ...

A join graph is used to define valid join paths between tables.
"""

import os
import json
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from utils import (
    Attribute, AttributeType, AttributeUsage, AttributeModality,
    DataStatistics, save_queries_to_file
)


# =============================================================================
# Join Path and Table Configuration
# =============================================================================

@dataclass
class JoinPath:
    """Represents a join relationship between two tables."""
    left_table: str
    right_table: str
    left_key: str
    right_key: str
    
    def to_sql_condition(self) -> str:
        """Return the SQL JOIN condition."""
        return f"{self.left_table}.{self.left_key} = {self.right_table}.{self.right_key}"


@dataclass
class TableConfig:
    """Configuration for a single table."""
    name: str
    csv_path: str
    attributes: List[Attribute]


# =============================================================================
# Join Graph Class
# =============================================================================

class JoinGraph:
    """
    Manages join relationships between tables and generates JOIN queries.
    """
    
    def __init__(self, tables: Dict[str, TableConfig], join_paths: List[JoinPath]):
        """
        Initialize the join graph.
        
        Args:
            tables: Dictionary mapping table names to TableConfig objects
            join_paths: List of JoinPath objects defining valid joins
        """
        self.tables = tables
        self.join_paths = join_paths
        self.data_stats = {}  # DataStatistics for each table
        
        # Load data statistics for each table
        for table_name, config in tables.items():
            if os.path.exists(config.csv_path):
                self.data_stats[table_name] = DataStatistics(config.csv_path)
    
    def get_join_path(self, table1: str, table2: str) -> Optional[JoinPath]:
        """Find a join path between two tables."""
        for jp in self.join_paths:
            if (jp.left_table == table1 and jp.right_table == table2):
                return jp
            if (jp.left_table == table2 and jp.right_table == table1):
                # Return reversed join path
                return JoinPath(table1, table2, jp.right_key, jp.left_key)
        return None
    
    def find_multi_table_path(self, tables: List[str]) -> List[JoinPath]:
        """
        Find a sequence of join paths connecting multiple tables.
        Uses simple BFS to find connected path.
        """
        if len(tables) < 2:
            return []
        
        path = []
        remaining = set(tables[1:])
        current = tables[0]
        visited = {current}
        
        while remaining:
            found = False
            for next_table in list(remaining):
                jp = self.get_join_path(current, next_table)
                if jp:
                    path.append(jp)
                    visited.add(next_table)
                    remaining.remove(next_table)
                    current = next_table
                    found = True
                    break
            
            if not found:
                # Try to find path through visited tables
                for v in visited:
                    for next_table in list(remaining):
                        jp = self.get_join_path(v, next_table)
                        if jp:
                            path.append(jp)
                            visited.add(next_table)
                            remaining.remove(next_table)
                            found = True
                            break
                    if found:
                        break
            
            if not found:
                raise ValueError(f"Cannot find join path for tables: {remaining}")
        
        return path
    
    def get_all_attributes(self, table_names: List[str] = None) -> List[Attribute]:
        """Get all attributes from specified tables (or all tables if None)."""
        if table_names is None:
            table_names = list(self.tables.keys())
        
        all_attrs = []
        for name in table_names:
            if name in self.tables:
                all_attrs.extend(self.tables[name].attributes)
        return all_attrs


# =============================================================================
# SELECT Clause Builder for JOIN Queries
# =============================================================================

def build_select_in_join_clause(
    join_graph: JoinGraph,
    table_names: List[str],
    select_attr_num: int = 4,
    seed: int = None
) -> Tuple[str, List[Attribute]]:
    """
    Build a SELECT clause for JOIN queries with table-prefixed attributes.
    
    Format: SELECT {table1}.{attr1}, {table2}.{attr2}, ...
    
    Args:
        join_graph: JoinGraph with table configurations
        table_names: List of table names involved in the join
        select_attr_num: Total number of attributes to select
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (SELECT clause string, list of selected attributes)
        e.g., ("SELECT player.name, team.team_name", [Attribute(...), ...])
    """
    if seed is not None:
        random.seed(seed)
    
    # Get attributes from all involved tables
    all_attrs = []
    for table in table_names:
        if table in join_graph.tables:
            all_attrs.extend(join_graph.tables[table].attributes)
    
    if not all_attrs:
        raise ValueError("No attributes available from specified tables")
    
    # Select at least one attribute from each table
    selected = []
    per_table = max(1, select_attr_num // len(table_names))
    
    for table in table_names:
        table_attrs = join_graph.tables[table].attributes
        if table_attrs:
            num_select = min(per_table, len(table_attrs))
            selected.extend(random.sample(table_attrs, num_select))
    
    # Add more if needed
    remaining = [a for a in all_attrs if a not in selected]
    while len(selected) < select_attr_num and remaining:
        attr = random.choice(remaining)
        selected.append(attr)
        remaining.remove(attr)
    
    # Shuffle to randomize order
    random.shuffle(selected)
    
    # Build SELECT clause with table prefixes
    select_parts = [f"{attr.table}.{attr.name}" for attr in selected]
    select_clause = f"SELECT {', '.join(select_parts)}"
    
    return select_clause, selected


# =============================================================================
# FROM JOIN Clause Builder
# =============================================================================

def build_from_join_clause(
    join_graph: JoinGraph,
    table_names: List[str]
) -> Tuple[str, List[str]]:
    """
    Build a FROM ... JOIN ... ON ... clause.
    
    Format (binary): FROM {table1} JOIN {table2} ON {join_condition}
    Format (multi):  FROM {table1} JOIN {table2} ON {cond1} JOIN {table3} ON {cond2}
    
    Args:
        join_graph: JoinGraph with table configurations
        table_names: List of table names to join (minimum 2)
    
    Returns:
        Tuple of (FROM clause string, list of join conditions)
        e.g., ("FROM player JOIN team ON player.team = team.team_name", [...])
    """
    if len(table_names) < 2:
        raise ValueError("JOIN requires at least 2 tables")
    
    from_table = table_names[0]
    join_conditions = []
    
    if len(table_names) == 2:
        # Binary join
        join_path = join_graph.get_join_path(table_names[0], table_names[1])
        if not join_path:
            raise ValueError(f"No join path between {table_names[0]} and {table_names[1]}")
        
        condition = join_path.to_sql_condition()
        join_conditions.append(condition)
        from_clause = f"FROM {from_table} JOIN {table_names[1]} ON {condition}"
    
    else:
        # Multi-table join
        join_paths = join_graph.find_multi_table_path(table_names)
        
        join_clauses = []
        joined_tables = {from_table}
        
        for jp in join_paths:
            condition = jp.to_sql_condition()
            join_conditions.append(condition)
            
            # Determine which table to add
            if jp.right_table not in joined_tables:
                join_clauses.append(f"JOIN {jp.right_table} ON {condition}")
                joined_tables.add(jp.right_table)
            elif jp.left_table not in joined_tables:
                # Reverse the join
                reversed_cond = f"{jp.right_table}.{jp.right_key} = {jp.left_table}.{jp.left_key}"
                join_clauses.append(f"JOIN {jp.left_table} ON {reversed_cond}")
                joined_tables.add(jp.left_table)
                join_conditions[-1] = reversed_cond
        
        from_clause = f"FROM {from_table} " + " ".join(join_clauses)
    
    return from_clause, join_conditions


# =============================================================================
# Full JOIN Query Generators
# =============================================================================

def generate_binary_join_query(
    join_graph: JoinGraph,
    table1: str,
    table2: str,
    select_attr_num: int = 4,
    seed: int = None
) -> Tuple[str, Dict]:
    """
    Generate a binary (2-table) JOIN query.
    
    Args:
        join_graph: JoinGraph with table configurations
        table1: First table name
        table2: Second table name
        select_attr_num: Number of attributes to select
        seed: Random seed
    
    Returns:
        Tuple of (SQL query string, metadata dict)
    """
    table_names = [table1, table2]
    
    # Build SELECT clause
    select_clause, selected_attrs = build_select_in_join_clause(
        join_graph, table_names, select_attr_num, seed
    )
    
    # Build FROM JOIN clause
    from_clause, join_conditions = build_from_join_clause(join_graph, table_names)
    
    # Combine into full SQL query
    sql_query = f"{select_clause} {from_clause};"
    
    metadata = {
        "category": "Join",
        "subcategory": "binary_join",
        "tables": table_names,
        "join_conditions": join_conditions,
        "selected_attributes": [f"{a.table}.{a.name}" for a in selected_attrs]
    }
    
    return sql_query, metadata


def generate_multi_table_join_query(
    join_graph: JoinGraph,
    tables: List[str],
    select_attr_num: int = 5,
    seed: int = None
) -> Tuple[str, Dict]:
    """
    Generate a multi-table (3+) JOIN query.
    
    Args:
        join_graph: JoinGraph with table configurations
        tables: List of table names to join (minimum 3)
        select_attr_num: Number of attributes to select
        seed: Random seed
    
    Returns:
        Tuple of (SQL query string, metadata dict)
    """
    if len(tables) < 3:
        raise ValueError("Multi-table join requires at least 3 tables")
    
    # Build SELECT clause
    select_clause, selected_attrs = build_select_in_join_clause(
        join_graph, tables, select_attr_num, seed
    )
    
    # Build FROM JOIN clause
    from_clause, join_conditions = build_from_join_clause(join_graph, tables)
    
    # Combine into full SQL query
    sql_query = f"{select_clause} {from_clause};"
    
    metadata = {
        "category": "Join",
        "subcategory": "multi_table_join",
        "tables": tables,
        "num_joins": len(join_conditions),
        "join_conditions": join_conditions,
        "selected_attributes": [f"{a.table}.{a.name}" for a in selected_attrs]
    }
    
    return sql_query, metadata


# =============================================================================
# Helper Functions
# =============================================================================

def attrs_from_json(attr_list: List[Dict]) -> List[Attribute]:
    """Convert JSON attribute list to Attribute objects."""
    type_map = {
        "str": AttributeType.STRING, 
        "int": AttributeType.INTEGER, 
        "float": AttributeType.FLOAT
    }
    usage_map = {
        "categorical": AttributeUsage.CATEGORICAL, 
        "numerical": AttributeUsage.NUMERICAL, 
        "general": AttributeUsage.GENERAL
    }
    modality_map = {
        "text": AttributeModality.TEXT, 
        "image": AttributeModality.IMAGE
    }
    
    return [
        Attribute(
            name=a["name"], 
            table=a["table"],
            value_type=type_map.get(a["value_type"], AttributeType.STRING),
            usage=usage_map.get(a["usage"], AttributeUsage.GENERAL),
            modality=modality_map.get(a["modality"], AttributeModality.TEXT),
            is_nullable=a.get("is_nullable", False),
            description=a.get("description", "")
        ) for a in attr_list
    ]


def create_player_join_graph(base_path: str) -> JoinGraph:
    """
    Create a JoinGraph for the Player dataset.
    
    Args:
        base_path: Base path to the Player dataset directory
    
    Returns:
        Configured JoinGraph object
    """
    # Load attributes for each table
    attrs_data = json.load(open(f"{base_path}/Med_attributes.json"))
    
    # Create table configurations
    tables = {
        "player": TableConfig("player", f"{base_path}/player.csv", attrs_from_json(attrs_data["player"])),
        "team": TableConfig("team", f"{base_path}/team.csv", attrs_from_json(attrs_data["team"])),
        "manager": TableConfig("manager", f"{base_path}/manager.csv", attrs_from_json(attrs_data["manager"])),
        "city": TableConfig("city", f"{base_path}/city.csv", attrs_from_json(attrs_data["city"])),
    }
    
    # Define join paths
    join_paths = [
        JoinPath("player", "team", "team", "team_name"),
        JoinPath("team", "city", "location", "city_name"),
        JoinPath("team", "manager", "ownership", "name"),
        JoinPath("manager", "team", "nba_team", "team_name"),
    ]
    
    return JoinGraph(tables, join_paths)


def create_med_join_graph(base_path: str) -> JoinGraph:
    """
    Create a JoinGraph for the Med (Medical) dataset.
    
    Tables: drug, disease, institution
    Join relationships:
        - drug.disease_name -> disease.disease_name (drug treats disease)
        - institution.research_diseases -> disease.disease_name (institution researches disease)
    
    Args:
        base_path: Base path to the Med dataset directory
    
    Returns:
        Configured JoinGraph object
    """
    # Load attributes for each table
    attrs_data = json.load(open(f"{base_path}/Med_attributes.json"))
    
    # Create table configurations
    tables = {
        "drug": TableConfig("drug", f"{base_path}/drug.csv", attrs_from_json(attrs_data["drug"])),
        "disease": TableConfig("disease", f"{base_path}/disease.csv", attrs_from_json(attrs_data["disease"])),
        "institution": TableConfig("institution", f"{base_path}/institution.csv", attrs_from_json(attrs_data["institution"])),
    }
    
    # Define join paths
    join_paths = [
        JoinPath("drug", "disease", "disease_name", "disease_name"),
        JoinPath("institution", "disease", "research_diseases", "disease_name"),
    ]
    
    return JoinGraph(tables, join_paths)


# =============================================================================
# Batch Generation and Save
# =============================================================================

def generate_and_save_join_queries(
    join_graph: JoinGraph,
    output_dir: str,
    binary_join_pairs: List[Tuple[str, str]],
    multi_join_combinations: List[List[str]],
    num_queries_per_combination: int = 10,
    binary_select_num: int = 4,
    multi_select_num: int = 5
):
    """
    Generate JOIN queries and save to files.
    
    Args:
        join_graph: JoinGraph with table configurations
        output_dir: Directory to save the query files
        binary_join_pairs: List of (table1, table2) pairs for binary joins
        multi_join_combinations: List of table lists for multi-table joins
        num_queries_per_combination: Number of queries per combination
        binary_select_num: Number of attributes for binary join SELECT
        multi_select_num: Number of attributes for multi-table join SELECT
    
    Returns:
        List of all generated query dictionaries
    """
    all_queries = []
    
    # Generate Binary Join Queries
    print("  Generating binary join queries...")
    for t1, t2 in binary_join_pairs:
        for i in range(num_queries_per_combination):
            sql, meta = generate_binary_join_query(
                join_graph, t1, t2,
                select_attr_num=binary_select_num,
                seed=i * 100 + hash(t1 + t2) % 1000
            )
            all_queries.append({"sql": sql, "metadata": meta})
        print(f"    {t1} ⟕ {t2}: {num_queries_per_combination} queries")
    
    # Generate Multi-Table Join Queries
    print("  Generating multi-table join queries...")
    for tables in multi_join_combinations:
        for i in range(num_queries_per_combination):
            try:
                sql, meta = generate_multi_table_join_query(
                    join_graph, tables,
                    select_attr_num=multi_select_num,
                    seed=i * 200 + hash(str(tables)) % 1000
                )
                all_queries.append({"sql": sql, "metadata": meta})
            except Exception as e:
                print(f"    ⚠️ Error for {tables}: {e}")
                break
        print(f"    {' ⟕ '.join(tables)}: {num_queries_per_combination} queries")
    
    # Save to JSON and SQL
    json_path = os.path.join(output_dir, "join_queries.json")
    sql_path = os.path.join(output_dir, "join_queries.sql")
    
    save_queries_to_file(all_queries, json_path, format="json")
    save_queries_to_file(all_queries, sql_path, format="sql")
    
    return all_queries


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    # Configuration
    base_path = "/data/dengqiyan/UDA-Bench/Query/Med"
    output_path = f"{base_path}/Join"
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Setup Player Dataset Join Graph
    join_graph = create_med_join_graph(base_path)
    
    # Test clause builders
    print("\n" + "=" * 70)
    print("Testing Clause Builders")
    print("=" * 70)
    
    # Test build_select_in_join_clause
    select_clause, selected = build_select_in_join_clause(
        join_graph, ["disease", "institution"], select_attr_num=4
    )
    print(f"\nSELECT clause: {select_clause}")
    
    # Test build_from_join_clause
    from_clause, conditions = build_from_join_clause(
        join_graph, ["drug", "disease"]
    )
    print(f"FROM clause: {from_clause}")
    print(f"Join conditions: {conditions}")
    
    # Define join combinations
    # binary_join_pairs = [
    #     ("player", "team"),
    #     ("team", "city"),
    #     ("team", "manager"),
    # ]
    
    # multi_join_combinations = [
    #     ["player", "team", "city"],
    #     ["player", "team", "manager"],
    #     ["team", "city", "manager"],
    #     ["player", "team", "city", "manager"],
    # ]

    binary_join_pairs = [
        ("drug", "disease"),
        # ("disease", "institution"),
    ]
    
    multi_join_combinations = [
        ["drug", "disease", "institution"],
    ]
    
    # Generate and save queries
    print("\n" + "=" * 70)
    print("GENERATING JOIN QUERIES")
    print("=" * 70)
    
    queries = generate_and_save_join_queries(
        join_graph=join_graph,
        output_dir=output_path,
        binary_join_pairs=binary_join_pairs,
        multi_join_combinations=multi_join_combinations,
        num_queries_per_combination=10,
        binary_select_num=4,
        multi_select_num=5
    )
    
    # Print summary
    print("\n" + "-" * 70)
    binary_count = sum(1 for q in queries if q['metadata']['subcategory'] == 'binary_join')
    multi_count = sum(1 for q in queries if q['metadata']['subcategory'] == 'multi_table_join')
    
    print(f"\n📊 Summary:")
    print(f"   Total queries: {len(queries)}")
    print(f"   Binary joins: {binary_count}")
    print(f"   Multi-table joins: {multi_count}")
    print(f"   Saved to: {output_path}/")
