import pandas as pd
import numpy as np
from quest.utils.log import print_log
import copy
from quest.conf import sqlconst

def merge_table(U : pd.DataFrame, V : pd.DataFrame, key):
    """
    Fill table V into table U using the specified primary key.
    If a cell in U already contains data, it is not overwritten.

    Parameters:
    U (pd.DataFrame): The primary DataFrame.
    V (pd.DataFrame): The secondary DataFrame to merge.
    key (str or list): The primary key.

    Returns:
    pd.DataFrame: The merged DataFrame.
    """

    #print_log("try merge----------------")
    #print_log(U)
    #print_log(V)

    for column in V.columns:
        if column not in U.columns:
            U[column] = None

    if key not in U.columns:
        U[key] = None
    if key not in V.columns:
        V[key] = None
    
    #print("try merge:\n", U, "\n", V)
    
    U = U.set_index(key, inplace=False)
    V = V.set_index(key, inplace=False)
    
    for Vindex, Vrow in V.iterrows():

        if Vindex not in U.index:
            # add line
            U.loc[Vindex] = [np.nan] * len(U.columns)

        for column in V.columns:
            if pd.isna(Vrow[column]) or Vrow[column] == '':
                continue
            if pd.isna(U.at[Vindex, column]) or U.at[Vindex, column] == '':
                U.at[Vindex, column] = Vrow[column]
    U.reset_index(inplace=True)

    return U

def keep_table(df : pd.DataFrame, resList, key):
    """
    keep the column [key], with resList
    """
    df_new = df[df[key].isin(resList)]
    return df_new

def check_missing_columns(df :pd.DataFrame, attributeList : list):
    missing_cols = [col for col in attributeList if col not in df.columns]
    for col in missing_cols:
        df[col] = None
    return df

def fill_cells(df : pd.DataFrame, attributeList : list, indList : list, key : str):
    L = len(indList)
    t = list(range(L))
    ndf = pd.DataFrame('None', columns=attributeList, index = t)
    ndf[key] = indList
    return merge_table(df, ndf, key)


def check_dict(d : dict, keyList : list):
    """
    check if d has keyList, if not, return a empty dict
    """
    res = {}
    for key in keyList:
        if key in d.keys():
            res[key] = d[key]
    return res

def check_dict_and_table(d: dict, doc_idList : list, columns : list[str], df : pd.DataFrame):
    """
    check if d has keyList, if not, return a empty dict
    also check if column has been extract

    d: the dict {doc_id : {column : [text , ]}}
    doc_idList : res_docList to be extract
    column : now filter column
    df : have accessed table
    """
    if df.empty:
        return check_dict(d, doc_idList)
    
    res = {}
    ndf = df.set_index('doc_id', inplace=False)
    #print("check_dict:\n", ndf)
    for doc_id in doc_idList:

        # not need to be extract
        if doc_id not in d.keys():
            continue

        for col in columns:

            # no such column
            if not col in d[doc_id].keys():
                continue
            
            if (doc_id in ndf.index) and (col in ndf.columns):
                val = ndf.at[doc_id, col]
                if pd.notna(val) and val != '':
                    # exist and have value, dont need to extract
                    continue
                else:
                    res.setdefault(doc_id, {})
                    res[doc_id][col] = d[doc_id][col]
            else:
                res.setdefault(doc_id, {})
                res[doc_id][col] = d[doc_id][col]

    #print("after check_dict:\n", res)

    return res


def aggregation_table(df: pd.DataFrame, columns: list, funcs: list, group_columns: list) -> pd.DataFrame:
    """
    Apply aggregation on DataFrame df based on specified parameters for multiple columns and multiple group columns.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns (list): List of column names on which aggregation is applied.
    funcs (list): List of aggregation functions to apply (e.g., MIN, MAX, SUM, AVG, COUNT).
    group_columns (list): List of column names to group by.

    Returns:
    pd.DataFrame: The aggregated DataFrame with renamed columns.
    """
    # Mapping functions to pandas aggregation functions
    func_mapping = {
        'MIN': 'min',
        'MAX': 'max',
        'SUM': 'sum',
        'AVG': 'mean',
        'COUNT': 'count'
    }

    # Check if all functions are valid
    for func in funcs:
        if func.upper() not in func_mapping:
            raise ValueError(f"Invalid aggregation function: {func}. Must be one of {list(func_mapping.keys())}")

    # Apply aggregation for each column and group combination
    aggr_dfs = []
    for column in columns:
        for func in funcs:
            aggr_func = func_mapping[func.upper()]
            grouped = df.groupby(group_columns)[column].agg(aggr_func).reset_index()
            grouped.rename(columns={column: f"{func.lower()}_{column}"}, inplace=True)
            aggr_dfs.append(grouped)

    # Merge all aggregated DataFrames on group columns
    final_df = aggr_dfs[0]
    for aggr_df in aggr_dfs[1:]:
        final_df = pd.merge(final_df, aggr_df, on=group_columns)

    return final_df


def custom_agg(x, func_name):
    if func_name in ['min', 'max', 'mean']:  
        # 对于 min 和 max，跳过缺失值
        return getattr(x, func_name)(skipna=True)
    else:
        return getattr(x, func_name)()
    
def count_all(x):
    return x.count()

def aggregation_table_transform(df: pd.DataFrame, functions : list[tuple], group_columns: list) -> pd.DataFrame:
    """
    Apply aggregation on DataFrame df based on specified parameters for multiple columns and multiple group columns.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    functions : list of tupe (column, func)
    columns :  column names on which aggregation is applied.
    funcs : aggregation functions to apply (e.g., MIN, MAX, SUM, AVG, COUNT).
    group_columns (list): List of column names to group by.

    Returns:
    pd.DataFrame: The DataFrame with original rows and added aggregation columns.
    """
    # Mapping functions to pandas aggregation functions
    func_mapping = {
        'MIN': 'min',
        'MAX': 'max',
        'SUM': 'sum',
        'AVG': 'mean',
        'COUNT': 'count'
    }

    # Check if all functions are valid
    for col, func in functions:
        if func.upper() not in func_mapping:
            raise ValueError(f"Invalid aggregation function: {func}. Must be one of {list(func_mapping.keys())}")

    # Create a copy of the original DataFrame to avoid modifying the input
    df_copy = df.copy()

    # For each column, apply the aggregation function using transform

    #print_log("before aggr df : \n",df_copy)
    df_copy = df_copy.dropna()
    func_map_columns = []

    for col in group_columns:
        df_copy[col] = df_copy[col].str.strip()

    for col, func in functions:
        aggr_func = func_mapping[func.upper()]

        if col == sqlconst.ALL_COLUMNS:
            # Apply count(*)
            df_copy["Count(*)"] =  df_copy.groupby(group_columns)['doc_id'].transform(count_all)
            func_map_columns.append("Count(*)")
        else:
            # Apply the aggregation function to each group and assign it back to the original DataFrame
            #print_log(f"apply {func} on {col} with group {group_columns}")
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
            df_copy[f"{func.lower()}_{col}"] = df_copy.groupby(group_columns)[col].transform(lambda x: custom_agg(x, aggr_func))
            func_map_columns.append(f"{func.lower()}_{col}")

    print_log("after aggr df : \n",df_copy)

    # Drop irrelevant columns (we keep only the group columns, the aggregation columns, and the relevant original ones)
    relevant_columns = group_columns + func_map_columns
    df_copy = df_copy[relevant_columns]

    #print_log("after drop aggr df : \n",df_copy)

    result = df_copy.drop_duplicates(subset=group_columns) # each group keep only one row

   # print_log("after keep one row df : \n",result)

    return result
