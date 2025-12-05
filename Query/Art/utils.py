"""
UDA-Bench Query Generation - Utility Classes and Functions

This module contains base classes, enums, and utility functions used across
all query generation modules.
"""

import os
import json
import random
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
from collections import Counter


# =============================================================================
# Enums for Attribute Metadata
# =============================================================================

class AttributeType(Enum):
    """Attribute value types"""
    STRING = "str"
    INTEGER = "int"
    FLOAT = "float"
    BOOLEAN = "bool"


class AttributeUsage(Enum):
    """Specific usage of attribute in queries"""
    CATEGORICAL = "categorical"    # For GROUP BY, categorical filters
    NUMERICAL = "numerical"        # For COUNT, SUM, AVG, MIN, MAX, numerical comparisons
    GENERAL = "general"            # For all usage


class AttributeModality(Enum):
    """Source modality of the attribute"""
    TEXT = "text"           # Plain text data
    IMAGE = "image"         # Image data (e.g., X-ray, photos)
    TABLE = "table"         # Table data
    STRUCTURED = "structured"  # Structured data from database


# =============================================================================
# Attribute Data Class
# =============================================================================

@dataclass
class Attribute:
    """
    Represents a database attribute with rich metadata.
    
    Attributes:
        name: Name of the attribute (column name)
        table: Table this attribute belongs to
        value_type: Data type of the attribute values (str, int, float, bool)
        usage: How this attribute is typically used in queries
        modality: Source modality of the data (text, image, table, etc.)
        is_nullable: Whether this attribute can have NULL values
        description: Optional description of the attribute
    """
    name: str
    table: str
    value_type: AttributeType
    usage: AttributeUsage
    modality: AttributeModality
    is_nullable: bool = False
    description: str = ""
    
    @property
    def full_name(self) -> str:
        """Return fully qualified name: table.attribute"""
        return f"{self.table}.{self.name}"

    def attribute_name(self) -> str:
        """Return the name of the attribute"""
        return self.name
    
    def is_groupable(self) -> bool:
        """Check if attribute can be used in GROUP BY"""
        return self.usage in [AttributeUsage.CATEGORICAL]
    
    def is_aggregatable(self) -> bool:
        """Check if attribute can be used with aggregation functions"""
        return self.usage == AttributeUsage.NUMERICAL
    
    def is_joinable(self) -> bool:
        """Check if attribute can be used as join key"""
        return self.usage == AttributeUsage.IDENTIFIER
    
    def supports_comparison(self) -> bool:
        """Check if attribute supports comparison operators (<, >, =, etc.)"""
        return self.value_type in [AttributeType.INTEGER, AttributeType.FLOAT, AttributeType.STRING]
    
    def supports_like(self) -> bool:
        """Check if attribute supports LIKE operator"""
        return self.value_type == AttributeType.STRING


# =============================================================================
# Attribute Loading Functions
# =============================================================================

def load_attributes_from_json(json_path: str) -> List[Attribute]:
    """
    Load attributes from a JSON file and convert them to Attribute objects.
    
    Args:
        json_path: Path to the JSON file containing attribute definitions
    
    Returns:
        List of Attribute objects
    """
    # Mapping from string values to Enum types
    type_mapping = {
        "str": AttributeType.STRING,
        "int": AttributeType.INTEGER,
        "float": AttributeType.FLOAT,
        "bool": AttributeType.BOOLEAN
    }
    
    usage_mapping = {
        "categorical": AttributeUsage.CATEGORICAL,
        "numerical": AttributeUsage.NUMERICAL,
        "general": AttributeUsage.GENERAL
    }
    
    modality_mapping = {
        "text": AttributeModality.TEXT,
        "image": AttributeModality.IMAGE,
        "table": AttributeModality.TABLE,
        "structured": AttributeModality.STRUCTURED
    }
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    attributes = []
    
    # Handle both flat list and nested dict formats
    if isinstance(data, dict):
        # Nested format: {"Wiki_Text": [...], "Wiki_Art": [...]}
        for source_key, attr_list in data.items():
            for attr_dict in attr_list:
                attr = Attribute(
                    name=attr_dict["name"],
                    table=attr_dict["table"],
                    value_type=type_mapping.get(attr_dict["value_type"], AttributeType.STRING),
                    usage=usage_mapping.get(attr_dict["usage"], AttributeUsage.GENERAL),
                    modality=modality_mapping.get(attr_dict["modality"], AttributeModality.TEXT),
                    is_nullable=attr_dict.get("is_nullable", False),
                    description=attr_dict.get("description", "")
                )
                attributes.append(attr)
    elif isinstance(data, list):
        # Flat format: [...]
        for attr_dict in data:
            attr = Attribute(
                name=attr_dict["name"],
                table=attr_dict["table"],
                value_type=type_mapping.get(attr_dict["value_type"], AttributeType.STRING),
                usage=usage_mapping.get(attr_dict["usage"], AttributeUsage.GENERAL),
                modality=modality_mapping.get(attr_dict["modality"], AttributeModality.TEXT),
                is_nullable=attr_dict.get("is_nullable", False),
                description=attr_dict.get("description", "")
            )
            attributes.append(attr)
    
    return attributes


# =============================================================================
# Data Statistics Class
# =============================================================================

class DataStatistics:
    """
    Class to load ground truth data and compute statistics for realistic literal generation.
    """
    
    def __init__(self, csv_path: str):
        """
        Load CSV data and compute statistics for each column.
        
        Args:
            csv_path: Path to the ground truth CSV file
        """
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'gbk', 'cp1252', 'iso-8859-1']
        for encoding in encodings:
            try:
                self.df = pd.read_csv(csv_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            # Fallback: read with errors='ignore'
            self.df = pd.read_csv(csv_path, encoding='utf-8', errors='ignore')
        self.total_rows = len(self.df)
        self.column_stats = {}
        self._compute_statistics()
    
    def _compute_statistics(self):
        """Compute value distribution statistics for each column."""
        # Define columns that use || as multi-value separator
        MULTI_VALUE_SEPARATOR = '||'
        
        for col in self.df.columns:
            # Skip non-filterable columns
            if col in ['id', 'Artwork_URL', 'intro_url']:
                continue
            
            values = self.df[col].dropna()
            
            if len(values) == 0:
                continue
            
            # Determine column type
            if pd.api.types.is_numeric_dtype(values):
                self.column_stats[col] = {
                    'type': 'numeric',
                    'is_multi_value': False,
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'mean': float(values.mean()),
                    'median': float(values.median()),
                    'percentiles': {
                        10: float(values.quantile(0.1)),
                        25: float(values.quantile(0.25)),
                        50: float(values.quantile(0.5)),
                        75: float(values.quantile(0.75)),
                        90: float(values.quantile(0.9))
                    },
                    'values': sorted(values.unique().tolist())
                }
            else:
                # String/categorical column - check for multi-value attributes
                str_values = values.astype(str)
                
                # Check if this column contains multi-value entries (using ||)
                has_multi_value = str_values.str.contains(r'\|\|', regex=True).any()
                
                if has_multi_value:
                    # Split multi-value entries and count individual values
                    all_individual_values = []
                    for val in str_values:
                        # Split by || and strip whitespace
                        parts = [p.strip() for p in val.split(MULTI_VALUE_SEPARATOR)]
                        all_individual_values.extend(parts)
                    
                    value_counts = Counter(all_individual_values)
                else:
                    # Single value column
                    value_counts = Counter(str_values)
                
                sorted_by_freq = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
                
                self.column_stats[col] = {
                    'type': 'categorical',
                    'is_multi_value': has_multi_value,
                    'unique_count': len(value_counts),
                    'value_counts': dict(sorted_by_freq),
                    # High frequency values (common, high selectivity)
                    'high_freq_values': [v for v, c in sorted_by_freq[:10]],
                    # Medium frequency values
                    'medium_freq_values': [v for v, c in sorted_by_freq[10:30] if c > 1],
                    # Low frequency values (rare, low selectivity)
                    'low_freq_values': [v for v, c in sorted_by_freq if c <= 2][:20],
                    'all_values': list(value_counts.keys())
                }
    
    def get_literal_by_selectivity(self, attr_name: str, selectivity: str = "medium") -> Tuple[str, float]:
        """
        Get a literal value based on the desired selectivity.
        
        Args:
            attr_name: Name of the attribute (must match column name, case-insensitive)
            selectivity: 'low' (rare values, few results), 
                        'medium' (moderate frequency),
                        'high' (common values, many results)
        
        Returns:
            Tuple of (literal_value, estimated_selectivity_ratio)
        """
        # Find matching column (case-insensitive)
        col_name = None
        for col in self.column_stats.keys():
            if col.lower() == attr_name.lower():
                col_name = col
                break
        
        if col_name is None:
            # Fallback to default values
            return "'unknown'", 0.0
        
        stats = self.column_stats[col_name]
        
        if stats['type'] == 'numeric':
            return self._get_numeric_literal(stats, selectivity)
        else:
            return self._get_categorical_literal(stats, selectivity)
    
    def _get_numeric_literal(self, stats: Dict, selectivity: str) -> Tuple[str, float]:
        """Generate numeric literal based on selectivity using percentiles."""
        percentiles = stats['percentiles']
        
        if selectivity == "low":
            # Use extreme values (top 10%) - fewer results when using < or >
            value = percentiles[90]
            ratio = 0.1
        elif selectivity == "high":
            # Use values around median - more results
            value = percentiles[50]
            ratio = 0.5
        else:  # medium
            # Use 25th or 75th percentile
            value = random.choice([percentiles[25], percentiles[75]])
            ratio = 0.25
        
        # Round to integer if all values are integers
        if all(isinstance(v, int) or (isinstance(v, float) and v.is_integer()) 
               for v in stats['values'][:10]):
            return str(int(value)), ratio
        else:
            return f"{value:.2f}", ratio
    
    def _get_categorical_literal(self, stats: Dict, selectivity: str) -> Tuple[str, float]:
        """Generate categorical literal based on selectivity using frequency."""
        value_counts = stats['value_counts']
        total = sum(value_counts.values())
        
        if selectivity == "low":
            # Pick rare values (low frequency)
            candidates = stats['low_freq_values']
            if not candidates:
                candidates = stats['all_values'][-10:]  # Last 10 values
        elif selectivity == "high":
            # Pick common values (high frequency)
            candidates = stats['high_freq_values']
        else:  # medium
            candidates = stats['medium_freq_values']
            if not candidates:
                candidates = stats['all_values']
        
        if not candidates:
            candidates = stats['all_values']
        
        value = random.choice(candidates)
        count = value_counts.get(value, 1)
        ratio = count / total
        
        # Escape single quotes in value
        escaped_value = value.replace("'", "''")
        return f"'{escaped_value}'", ratio
    
    def get_column_info(self, col_name: str) -> Optional[Dict]:
        """Get statistics for a specific column."""
        for col in self.column_stats.keys():
            if col.lower() == col_name.lower():
                return self.column_stats[col]
        return None


# =============================================================================
# File I/O Functions
# =============================================================================

def save_queries_to_file(queries: List[Dict], output_path: str, format: str = "json"):
    """
    Save generated queries to a file.
    
    Args:
        queries: List of query dictionaries with 'sql' and 'metadata' keys
        output_path: Path to save the file
        format: Output format ('json' or 'sql')
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    if format == "json":
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(queries, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved {len(queries)} queries to {output_path}")
        
    elif format == "sql":
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, q in enumerate(queries):
                subcategory = q.get('metadata', {}).get('subcategory', 'unknown')
                tables = q.get('metadata', {}).get('tables', [])
                f.write(f"-- Query {i+1}: {subcategory} ({', '.join(tables) if tables else 'N/A'})\n")
                f.write(q['sql'] + "\n\n")
        print(f"✅ Saved {len(queries)} queries to {output_path}")
    
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'sql'.")

