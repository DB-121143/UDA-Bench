import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from litellm import batch_completion
from utils import read_and_concatenate, read_image
from config_uqe import MODEL, OPENAI_KEY, BASE_URL


client = OpenAI(api_key=OPENAI_KEY, base_url=BASE_URL)


def func(x, cond=None):
    return random.randint(0, 1)

def all_one(x, cond=None):
    if x == None:
        return 0
    return 1

def df_check_nan(df_row):
    if isinstance(df_row, pd.Series):
        if df_row.count() == 0:
            return 0
    elif pd.isna(df_row):
        return 0
    return 1

def df_check_nan_sum(df_row):
    if pd.isna(df_row):
        return 0
    else:
        return df_row

def llm_filter(x, sys_prompt, model=MODEL, attempts=5):
    print("llm_filter")
    user_prompt = x
    result = ""
    for attempt in range(attempts):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system", 
                        "content": sys_prompt},
                    {
                        "role": "user", 
                        "content": user_prompt},
                ],
                temperature=0.1 + attempt * 0.1,
                max_tokens=50  # 返回的最大字数
            )
            
            result = response.choices[0].message.content
            if result == "True" or result == "**True**" or "True" in result:
                return 1
            elif "False" in result:
                return 0
            else:
                return random.randint(0, 1)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
    return random.randint(0, 1)


def llm_filter_merged(x, df_cols, sys_prompt, model=MODEL, attempts=5):
    llm_messages = []
    llm_messages.append(
        {"role": "system", "content": sys_prompt}
    )
    for col in df_cols:
        content = x[col]
        if "jpg" in content:
            content = read_image(None, content)
            llm_messages.append({"role": "user", "content": [
                {"type": "text", "text": "The image of '" + col + "' will be listed below. "},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{content}"}}
            ]})
        else:
            llm_messages.append(
                {"role": "user", "content": "The paragraph of '" + col + "' will be listed below. "}
            )
            llm_messages.append(
                {"role": "user", "content": content}
            )
    
    result = ""
    for attempt in range(attempts):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=llm_messages,
                temperature=0.1 + attempt * 0.1,
                max_tokens=100
            )
            
            result = response.choices[0].message.content
            # print(result)
            if result == "True" or result == "**True**" or "True" in result:
                return 1
            elif "False" in result:
                return 0
            else:
                return random.randint(0, 1)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")

    return random.randint(0, 1)

def llm_filter_merged_batch(df_X, df_cols, indices, sys_prompt, model=MODEL, attempts=8, batch_size=6):
    true_labels_list = []
    
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i + batch_size]
        
        batch_results = _process_single_batch(df_X, df_cols, batch_indices, sys_prompt, model, attempts)
        true_labels_list.extend(batch_results)
    
    return true_labels_list

def _process_single_batch(df_X, df_cols, batch_indices, sys_prompt, model=MODEL, attempts=8):
    
    # build messages list for current batch
    messages_list = []
    for idx in batch_indices:
        messages = []
        messages.append({"role": "system", "content": sys_prompt})
        for col in df_cols:
            messages.append({"role": "user", "content": "The paragraph of '" + col + "' will be listed below. "})
            # rough handling for image, need to be improved here
            content = df_X.iloc[idx][col]
            if "jpg" in content:
                filepath = content
                content = read_image(None, filepath)
                messages.append({"role": "user", "content": [
                    {
                        "type": "text",
                        "text": "The image of '" + col + "' will be listed below. "
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{content}"
                        }
                    }
                ]})
            else:
                messages.append({"role": "user", "content": content})
        messages_list.append(messages)
    
    for attempt in range(attempts):
        try:
            
            batch_params = {
                "model": "openai/" + model,
                "messages": messages_list,
                "temperature": 0.1 + attempt * 0.1,
                "max_tokens": 100,
                "api_key": OPENAI_KEY,
                "api_base": BASE_URL
            }
            
            responses = batch_completion(**batch_params)
            
            batch_results = []
            for response in responses:
                if response and hasattr(response, 'choices') and response.choices:
                    result = response.choices[0].message.content
                    
                    if result == "True" or result == "**True**" or "True" in result:
                        batch_results.append(1)
                    elif "False" in result:
                        batch_results.append(0)
                    else:
                        batch_results.append(random.randint(0, 1))
                else:
                    batch_results.append(random.randint(0, 1))
            
            # 如果所有请求都成功，返回结果
            if len(batch_results) == len(batch_indices):
                # print(f"Batch completed successfully with {len(batch_results)} results")
                return batch_results
                
        except Exception as e:
            print(f"Batch attempt {attempt + 1} failed: {e}")
            if attempt == attempts - 1:
                print(f"Falling back to individual processing for batch of {len(batch_indices)} samples...")
                batch_results = []
                for idx in batch_indices:
                    result = llm_filter_merged(df_X.iloc[idx], df_cols, sys_prompt, model, 1)
                    batch_results.append(result)
                return batch_results
    
    # 如果所有尝试都失败，返回随机值
    print(f"All attempts failed for batch, using random values")
    return [random.randint(0, 1) for _ in batch_indices]


def llm_filter_aggr_merged_split(user_prompt_dict, user_prompt_cell_list, filter_sys_prompt_split, model=MODEL, attempts=5):
    print("llm_filter_aggr_merged_split")
    res = False
    for sys_prompt in filter_sys_prompt_split:
        llm_messages = []
        llm_messages.append(
            {"role": "system", "content": sys_prompt}
        )
        for user_prompt_cell in user_prompt_cell_list:
            llm_messages.append(user_prompt_cell)
    
        result = ""
        for attempt in range(attempts):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=llm_messages,
                    temperature=0.1 + attempt * 0.1,
                    max_tokens=100
                )
                
                result = response.choices[0].message.content
                print(result)
                if result == "True" or result == "**True**" or "True" in result:
                    return 1
                else:
                    break
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
    return 0


def llm_extractor(df, col, attrs, sys_prompt, data_schema, df_id=None,
                  model=MODEL, retries=10):
    col_type = data_schema.get_col_type(col)
    new_df = pd.DataFrame(columns=[f"{col}.{attr}" for attr in attrs])
    for attr in attrs:
        attr_type = data_schema.get_attr_type(col, attr)
        if attr_type == "varchar":
            attr_type = "str"
        elif attr_type == "int":
            attr_type = "int"
        elif attr_type == "float":
            attr_type = "float"
        else:
            # date
            attr_type = "str"
        new_df[f"{col}.{attr}"] = new_df[f"{col}.{attr}"].astype(attr_type)

    for i in tqdm(range(len(df))):
        if col_type == "csv":
            filepath = df.iloc[i]
            cell = read_and_concatenate(data_schema, filepath)
        elif col_type == "image":
            filepath = df.iloc[i]
            cell = read_image(data_schema, filepath)
        else: # "text"
            cell = df.iloc[i]
        
        if col_type != "image":
            user_prompt_cell = {
                "role": "user",
                "content": "The paragraph of '" + col + "': " + cell
            }
        else:
            user_prompt_cell = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "The image of '" + col + "' will be listed below. "
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{cell}"
                        }
                    }
                ]
            }

        remain_attrs = attrs.copy()

        for attempt in range(retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system", 
                            "content": sys_prompt
                        },
                        user_prompt_cell,
                    ],
                    temperature=0.1 + attempt * 0.05,
                    max_tokens=300
                )
                
                results = response.choices[0].message.content
                # print(results)
                split_results = results.split("&&")
                for attr in remain_attrs:
                    found = False
                    for result in split_results:
                        result_attr = result.split(":")[0].strip()
                        result_val = result.split(":")[1].strip()
                        if result_attr == attr or result_attr == col + '.' + attr:
                            found = True
                            if result_val == "None":
                                new_df.loc[i, f"{col}.{attr}"] = pd.NA
                            else:
                                attr_type = data_schema.get_attr_type(col, attr)
                                if attr_type == "varchar":
                                    new_df.loc[i, f"{col}.{attr}"] = str(result_val)
                                elif attr_type == "int":
                                    try:
                                        new_df.loc[i, f"{col}.{attr}"] = int(result_val)
                                    except:
                                        new_df.loc[i, f"{col}.{attr}"] = result_val
                                elif attr_type == "float":
                                    try:
                                        new_df.loc[i, f"{col}.{attr}"] = float(result_val)
                                    except:
                                        new_df.loc[i, f"{col}.{attr}"] = result_val
                                elif attr_type == "date":
                                    try:
                                        new_df.loc[i, f"{col}.{attr}"] = str(result_val)
                                    except:
                                        new_df.loc[i, f"{col}.{attr}"] = result_val
                    if not found:
                        new_df.loc[i, f"{col}.{attr}"] = pd.NA
                break
            except Exception as e:
                print(f"Error processing row {i + 1}: {e}")

    return new_df

def llm_groupby_labels(df, col_name, system_prompt, data_schema,
                       model=MODEL, attempts=5):
    col_type = data_schema.get_col_type(col_name)
    sys_prompt_unit = {
        "role": "system",
        "content": system_prompt
    }
    messages = [sys_prompt_unit]
    for i in range(len(df)):
        if col_type == "csv":
            filepath = df.iloc[i]
            cell = read_and_concatenate(data_schema, filepath)
        elif col_type == "image":
            filepath = df.iloc[i]
            cell = read_image(data_schema, filepath)
        else: # "text"
            cell = df.iloc[i]
        
        if col_type != "image":
            user_prompt_unit = {
                "role": "user",
                "content": cell
            }
        else:
            user_prompt_unit = {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{cell}"
                        }
                    }
                ]
            }
        messages.append(user_prompt_unit)

    for attempt in range(attempts):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1 + attempt * 0.1,
                max_tokens=200
            )
            
            results = response.choices[0].message.content
            labels = results.split("||")
            labels = [label.strip() for label in labels]
            labels = list(set(labels))
            return labels
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
    return []

def llm_groupby_labels_split(df, col_name, system_prompt, data_schema,
                       model=MODEL, attempts=5):
    col_type = data_schema.get_col_type(col_name)
    sys_prompt_unit = {
        "role": "system",
        "content": system_prompt
    }

    labels_set = set()
    for i in range(len(df)):
        messages = [sys_prompt_unit]
        if col_type == "csv":
            filepath = df.iloc[i]
            cell = read_and_concatenate(data_schema, filepath)
        elif col_type == "image":
            filepath = df.iloc[i]
            cell = read_image(data_schema, filepath)
        else: # "text"
            cell = df.iloc[i]
        
        if col_type != "image":
            user_prompt_unit = {
                "role": "user",
                "content": "The paragraph of '" + col_name + "': " + cell
            }
        else:
            user_prompt_unit = {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": "The image of '" + col_name + "' will be listed below. "
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{cell}"
                        }
                    }
                ]
            }
        
        messages.append(user_prompt_unit)

        for attempt in range(attempts):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.1 + attempt * 0.1,
                    max_tokens=200
                )

                results = response.choices[0].message.content
                labels_set.add(results.lower().strip())
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
    
    return list(labels_set)

def llm_cluster(df, col_name_list, labels_dict, sys_prompt, data_schema, candidate_groups,
    model=MODEL, attempts=10):
    labels_key = list(labels_dict.keys())
    cluster_result = []
    
    for i in range(len(df)):
        llm_messages = []
        llm_messages.append(
            {"role": "system", "content": sys_prompt}
        )
        row = df.iloc[i]
        for col_name in col_name_list:
            col_type = data_schema.get_col_type(col_name)
            if col_type == "csv":
                filepath = row[col_name]
                cell = read_and_concatenate(data_schema, filepath)
            elif col_type == "image":
                filepath = row[col_name]
                cell = read_image(data_schema, filepath)
            else: # "text"
                cell = row[col_name]

            if col_type != "image":
                llm_messages.append(
                    {"role": "user", "content": "The paragraph of '" + col_name + "' will be listed below. "}
                )
                llm_messages.append(
                    {"role": "user", "content": cell}
                )
            else:
                llm_messages.append(
                    {"role": "user", "content": "The image of '" + col_name + "' will be listed below. "}
                )
                llm_messages.append(
                    {
                        "role": "user", 
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{cell}"
                                }
                            }
                        ]
                    }
                )

        success = False
        for attempt in range(attempts):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=llm_messages,
                    temperature=0.1 + attempt * 0.1,
                    max_tokens=300
                )
                
                results = response.choices[0].message.content
                split_results = results.split("&&")
                row_result = {}
                assert len(split_results) == len(labels_key)

                for j, split_result in enumerate(split_results):
                    attr_name = split_result.split(':')[0].strip()
                    attr_val = split_result.split(':')[1].strip()
                    row_result[attr_name] = attr_val
                if row_result not in candidate_groups:
                    print("row_result not in candidate_groups")
                    candidate_groups.append(row_result)
                cluster_result.append(row_result)
                success = True
                break
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
        
        if not success:
            print(f"All attempts failed for row {i}, using default value")
            if candidate_groups:
                cluster_result.append(candidate_groups[0])
            else:
                default_result = {}
                for key in labels_key:
                    default_result[key] = "None"
                cluster_result.append(default_result)
    
    return cluster_result
