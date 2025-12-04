"""
UDA-Bench Query Generation - SELECT Queries

This module generates SELECT queries following the template:
    SELECT {attribute(s)} FROM {table}

If image modality attributes exist, at least one must be included.
"""

import random
from typing import List, Tuple

from utils import Attribute, AttributeModality


# =============================================================================
# SELECT Clause Builder
# =============================================================================

def build_select_clause(
    attributes: List[Attribute], 
    attribute_num: int = 1,
    image_num: int = 1,
    seed: int = None,
    table: str = None
) -> Tuple[str, List[Attribute], str]:
    """
    Build a SELECT clause by selecting random attributes from the same table.
    
    If any IMAGE modality attribute exists in the available attributes,
    at least one image attribute MUST be included in the selection.
    
    Args:
        attributes: List of available Attribute objects
        attribute_num: Total number of attributes to select
        image_num: Number of image attributes to include (if available)
        seed: Random seed for reproducibility (optional)
        table: If specified, only select attributes from this table.
               If None, will pick a random table from available attributes.
    
    Returns:
        Tuple of (SELECT clause string, list of selected attributes, table name)
        e.g., ("SELECT name, age, photo", [Attribute(...), ...], "player")
    """
    if seed is not None:
        random.seed(seed)
    
    if not attributes:
        raise ValueError("Attribute list cannot be empty")
    
    # Get unique tables from attributes
    available_tables = list(set(attr.table for attr in attributes if attr.table))
    
    if not available_tables:
        # No table info in attributes, use all attributes
        selected_table = table if table else "unknown"
    elif table is not None:
        # Try exact match first
        if table in available_tables:
            selected_table = table
        else:
            # Try partial match (e.g., "Player" matches "Player_disease")
            matching_tables = [t for t in available_tables if table in t or t in table]
            if matching_tables:
                selected_table = matching_tables[0]
            else:
                selected_table = available_tables[0]
        attributes = [attr for attr in attributes if attr.table == selected_table]
    else:
        # Pick a random table
        selected_table = random.choice(available_tables)
        attributes = [attr for attr in attributes if attr.table == selected_table]
    
    if not attributes:
        raise ValueError(f"No attributes found for table: {selected_table}")
    
    # Separate image attributes from non-image attributes
    image_attrs = [attr for attr in attributes if attr.modality == AttributeModality.IMAGE]
    non_image_attrs = [attr for attr in attributes if attr.modality != AttributeModality.IMAGE]
    
    selected_attrs = []
    
    # If image attributes exist, must include at least one
    if image_attrs and image_num > 0:
        selected_image_attrs = random.sample(image_attrs, min(image_num, len(image_attrs)))
        selected_attrs.extend(selected_image_attrs)
    
    # Calculate remaining slots for non-image attributes
    remaining_slots = attribute_num - len(selected_attrs)
    
    # Fill remaining slots with non-image attributes
    if non_image_attrs and remaining_slots > 0:
        num_non_image = min(remaining_slots, len(non_image_attrs))
        selected_non_image_attrs = random.sample(non_image_attrs, num_non_image)
        selected_attrs.extend(selected_non_image_attrs)
    
    # Shuffle to randomize order
    random.shuffle(selected_attrs)
    
    # Build the SELECT clause
    attribute_names = ", ".join([attr.name for attr in selected_attrs])
    select_clause = f"SELECT {attribute_names}"
    
    return select_clause, selected_attrs, selected_table


# =============================================================================
# Full SELECT Query Generator
# =============================================================================

def generate_select_query(
    attributes: List[Attribute], 
    table: str, 
    attribute_num: int = 1,
    image_num: int = 1,
    seed: int = None
) -> Tuple[str, List[Attribute], str]:
    """
    Generate a complete SELECT query.
    
    Args:
        attributes: List of available Attribute objects
        table: Table name hint (will match to actual table in attributes)
        attribute_num: Total number of attributes to select
        image_num: Number of image attributes to include (if available)
        seed: Random seed for reproducibility (optional)
    
    Returns:
        Tuple of (SQL query string, list of selected attributes, actual table name)
    """
    # Build SELECT clause - only use attributes from the specified table
    select_clause, selected_attrs, actual_table = build_select_clause(
        attributes, attribute_num, image_num, seed, table=table
    )
    
    # Generate the complete SQL query using the actual table name from attributes
    sql_query = f"{select_clause} FROM {actual_table};"
    
    return sql_query, selected_attrs, actual_table


# =============================================================================
# Batch Generation and Save
# =============================================================================

def generate_and_save_select_queries(
    attributes: List[Attribute],
    table: str,
    output_dir: str,
    num_queries: int = 10,
    attribute_num: int = 3,
    image_num: int = 0
):
    """
    Generate multiple SELECT queries and save to file.
    
    Args:
        attributes: List of available Attribute objects
        table: Table name hint (will match to actual table in attributes)
        output_dir: Directory to save the query files
        num_queries: Number of queries to generate
        attribute_num: Number of attributes per query
        image_num: Number of image attributes to include
    
    Returns:
        List of generated query dictionaries
    """
    from utils import save_queries_to_file
    import os
    
    all_queries = []
    
    for i in range(num_queries):
        max_attribute_num = attribute_num
        current_query_attribute_num = random.randint(1, max_attribute_num)
        sql, selected_attrs, actual_table = generate_select_query(
            attributes, table, current_query_attribute_num, image_num, seed=i * 42
        )
        all_queries.append({
            "sql": sql,
            "metadata": {
                "category": "Select",
                "subcategory": "select",
                "selected_attributes": [a.name for a in selected_attrs],
                "tables": [actual_table]
            }
        })
    
    file_json_name = f"select_queries_{table}.json"
    file_sql_name = f"select_queries_{table}.sql"
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
    base_path = "/data/dengqiyan/UDA-Bench/Query/Med"
    attributes_path = f"{base_path}/Med_attributes.json"

    output_path = f"{base_path}/Select"
    
    # Load attributes
    attributes = load_attributes_from_json(attributes_path)
    print(f"✅ Loaded {len(attributes)} attributes")
    
    # Generate and save SELECT queries
    print("\n" + "=" * 70)
    print("GENERATING SELECT QUERIES")
    print("=" * 70)
    
    queries = generate_and_save_select_queries(
        attributes=attributes,
        table="disease",
        output_dir=output_path,
        num_queries=4,
        attribute_num=5,
        image_num=1
    )
    
    # Print examples
    print("\nGenerated queries:")
    print("-" * 70)
    for i, q in enumerate(queries[:5]):
        print(f"  {i+1}. {q['sql']}")
    if len(queries) > 5:
        print(f"  ... and {len(queries) - 5} more")
    
    print(f"\n📊 Summary:")
    print(f"   Total queries: {len(queries)}")
    print(f"   Saved to: {base_path}/")
