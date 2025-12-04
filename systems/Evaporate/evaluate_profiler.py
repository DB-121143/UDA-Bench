from collections import defaultdict, Counter
import numpy as np  # ENHANCED: Required for advanced statistical calculations
import json  # NEW ADDITION: For loading table.json files
import os  # NEW ADDITION: For file path operations
from evaporate.prompts import (PICK_VALUE_CONTEXT, Step,)
from evaporate.utils import apply_prompt


def clean_comparison(responses, field):
    clean_responses = []
    # ENHANCED VERSION: Major improvements in handling empty values and edge cases
    # print(f"clean_comparison input: {responses}")  # 注释掉调试输出
    # 处理各种空值情况
    if responses is None:
        # print("clean_comparison: input is None")
        return ""
    if isinstance(responses, str):
        # print("clean_comparison: input is string")
        if not responses.strip():
            return ""
        responses = [responses]
    elif isinstance(responses, list):
        # print("clean_comparison: input is list")
        # 扁平化嵌套列表
        flattened = []
        for item in responses:
            if isinstance(item, list):
                flattened.extend(x for x in item if x and str(x).strip())
            elif item and str(item).strip():
                flattened.append(str(item).strip())
        responses = flattened
        if not responses:
            # print("clean_comparison: flattened list is empty")
            return ""
    # OLD VERSION:
    # if type(responses) == str:
    #     responses = [responses]
    for response in responses:
        if not response or response is None:  # 跳过空值
            continue
        response = str(response).lower()
        if response.strip() in ['none', 'null', 'n/a', 'not available', 'not specified', '']:
            continue
        field_lower = field.lower()
        field_reformat = field_lower.replace("_", "-")
        # 对于name字段，不要移除太多内容
        if field_lower == "name":
            # 只移除明显的噪音字符，保留名字本身
            for char in ["'", ":", "<", ">", '"', "\n", "\t", "\r"]:
                response = response.replace(char, " ")
        else:
            # 对其他字段进行正常的清理
            for char in ["'", field_lower, field_reformat, ":", "<", ">", '"']:
                response = response.replace(char, " ")
        # 清理标点符号和多余空格
        for char in [",", ".", "?", "!", ";", "(", ")", "[", "]", "{", "}", "-"]:
            response = response.replace(char, " ")
        # 规范化空格
        response = " ".join(response.split())
        # 只有非空且不是无效值时才添加
        if response.strip() and response.lower() not in ['none', 'null', 'n/a', 'not available', 'not specified']:
            clean_responses.append(response.strip())
    result = ", ".join(clean_responses) if clean_responses else ""
    return result
    
    # OLD VERSION:
    # for response in responses:
    #     response = response.lower()
    #     field = field.lower()
    #     field_reformat = field.replace("_", "-")
    #     for char in ["'", field, field_reformat, ":", "<", ">", '"', "none"]:
    #         response = response.replace(char, " ")
    #     for char in [",", ".", "?", "!", ";", "(", ")", "[", "]", "{", "}", "-", "none", "\n", "\t", "\r"]: 
    #         response = response.replace(char, " ")
    #     response = response.replace("  ", " ")
    #     response = response.split()
    #     response = [r.strip() for r in response]
    #     response = [r for r in response if r]
    #     response = ' '.join(response)
    #     clean_responses.append(response)
    # clean_responses = ", ".join(clean_responses)
    # return clean_responses


def normalize_value_type(metadata, attribute):
    # ENHANCED VERSION: Major improvements in handling complex nested data structures and edge cases
    # make everything a list of strings since functions can return diverse types
    cleaned_items = [] 
    
    # 处理完全空值的情况
    if metadata is None or metadata == [] or metadata == [[]]:
        return []
    
    # OLD VERSION:
    # if type(metadata) == str:
    #     metadata = [metadata]
    # for item in metadata:
    #     if type(item) == list:
    #         item = [str(i) for i in item]
    #         item = ", ".join(item)
    #     elif type(item) == tuple:
    #         item = list(item)
    #         item = [str(i) for i in item] 
    #         item = ", ".join(item)
    #     elif item is None:
    #         item = ''
    #     elif type(item) != str:
    #         item = [str(item)]
    #         item = ", ".join(item)
    #     if item: 
    #         cleaned_items.append(item)
    # return cleaned_items
    
    if isinstance(metadata, str):
        # 检查是否包含 || 分隔符
        if "||" in metadata:
            metadata = [item.strip() for item in metadata.split("||") if item.strip()]
        else:
            metadata = [metadata]
    elif isinstance(metadata, list):
        # 处理空列表和[None]等特殊情况
        if not metadata or all(x is None or x == '' or x == [] for x in metadata):
            return []
            
        # 处理 [[value]] 格式
        if metadata and isinstance(metadata[0], list):
            if not metadata[0] or all(x is None or x == '' for x in metadata[0]):
                return []
            # 把所有非空的值都添加到结果中
            valid_values = []
            for sublist in metadata:
                if isinstance(sublist, list) and sublist:
                    for item in sublist:
                        if item and str(item).strip():
                            valid_values.append(str(item).strip())
            return valid_values if valid_values else []
            
        # 如果是普通列表，继续处理
        metadata = [str(x).strip() for x in metadata if x is not None and str(x).strip()]
    
    for item in metadata:
        if isinstance(item, list):
            # 处理嵌套列表
            if not item:  # 空列表
                continue
            item = [str(i) for i in item if i is not None and str(i).strip()]
            item = ", ".join(item)
        elif isinstance(item, tuple):
            item = list(item)
            item = [str(i) for i in item if i is not None and str(i).strip()]
            item = ", ".join(item)
        elif item is None or item == '':
            continue
        elif not isinstance(item, str):
            item = str(item)
        
        # 对字符串项检查是否包含 || 分隔符
        if isinstance(item, str) and "||" in item:
            sub_items = [sub_item.strip() for sub_item in item.split("||") 
                        if sub_item and sub_item.strip()]
            cleaned_items.extend(sub_items)
            continue
            
        if item and str(item).strip():  # 只添加非空的项目
            cleaned_items.append(str(item).strip())
    
    # 过滤掉空值、None等
    cleaned_items = [item for item in cleaned_items 
                    if item and item.lower() not in ['none', 'null', 'n/a', '']]
    
    return cleaned_items


def pick_a_gold_label(golds, attribute="", manifest_session=None, overwrite_cache=False):
    """
    To counteract the large model hallucinating on various chunks affecting the evaluation of good functions.
    """

    pred_str = "- " + "\n- ".join(golds)

    prompt_template = PICK_VALUE_CONTEXT[0]
    prompt = prompt_template.format(pred_str=pred_str, attribute=attribute)
    try:
        check, num_toks = apply_prompt(
            Step(prompt), 
            max_toks=100, 
            manifest=manifest_session,
            overwrite_cache=overwrite_cache
        )
    except:
        return golds, 0 
    check = check.split("\n")
    check = [c for c in check if c]
    if check:
        if "none" in check[0].lower():
            check = golds
        else:
            check = check[0]
    return check, num_toks


# ...existing code...

def text_f1(
    preds=[], 
    golds=[], 
    extraction_fraction=1.0, 
    attribute=None,
    extraction_fraction_thresh=0.8,
    use_abstension=True,
    function_name=None,  # 新增参数
):
    """Compute average F1 of text spans.
    Taken from Squad without prob threshold for no answer.
    过滤掉标准答案为空的样本以避免影响函数性能评估
    """
    # ENHANCED VERSION: Major improvements in evaluation logic, debugging, and edge case handling
    # Added comprehensive filtering and debugging capabilities
    total_f1 = 0
    total_recall = 0
    total_prec = 0
    f1s = []
    total = 0

    # 首先过滤掉标准答案为空的样本
    filtered_preds = []
    filtered_golds = []
    
    for pred, gold in zip(preds, golds):
        # 检查gold是否为空或无意义
        gold_is_empty = False
        if isinstance(gold, str):
            # 标准化检查空值
            gold_cleaned = gold.strip().lower()
            if (not gold_cleaned or 
                gold_cleaned in ['', 'none', 'null', 'n/a', 'not available', 'not specified']):
                gold_is_empty = True
        elif isinstance(gold, list):
            # 对于列表类型的gold，检查是否所有元素都为空
            if not gold or all(not str(g).strip() for g in gold):
                gold_is_empty = True
        
        # 只保留标准答案非空的样本
        if not gold_is_empty:
            filtered_preds.append(pred)
            filtered_golds.append(gold)
    
    # 如果过滤后没有有效样本，返回0分
    if not filtered_preds:
        print(f"Warning: All gold labels are empty for attribute '{attribute}', skipping evaluation")
        return 0.0, 0.0
    
    # 使用过滤后的数据进行原有的评估逻辑
    preds = filtered_preds
    golds = filtered_golds
    
    # ===== 添加调试信息开始 =====
    print(f"\n=== text_f1 调试信息 - {attribute}{function_name} ===")
    print(f"输入preds数量: {len(preds)}")
    print(f"输入golds数量: {len(golds)}")
    
    for i, (pred, gold) in enumerate(zip(preds[:10], golds[:10])):  # 只显示前10个
        print(f"样本{i}: pred={pred}, gold={gold}")
    # ===== 调试信息结束 =====

    # OLD VERSION LOGIC (commented out):
    # if extraction_fraction >= extraction_fraction_thresh and use_abstension:
    #     new_preds = []
    #     new_golds = []
    #     for pred, gold in zip(preds, golds):
    #         if pred:
    #             new_preds.append(pred)
    #             new_golds.append(gold)
    #     preds = new_preds
    #     golds = new_golds
    #     if not preds:
    #         return 0.0, 0.0
    # for pred, gold in zip(preds, golds):
    #     if type(pred) == str:
    #         pred_toks = pred.split()
    #     else:
    #         pred_toks = pred
    #     if type(gold) == str:
    #         gold_toks_list = [gold.split()]
    #     else:
    #         assert 0, print(gold)
    #         gold_toks_list = gold

    # if extraction_fraction >= extraction_fraction_thresh and use_abstension:
    #     new_preds = []
    #     new_golds = []
    #     for pred, gold in zip(preds, golds):
    #         if pred:
    #             new_preds.append(pred)
    #             new_golds.append(gold)
    #     preds = new_preds
    #     golds = new_golds
    #     if not preds:
    #         return 0.0, 0.0
    
    # ===== 再次添加调试信息 =====
    if extraction_fraction >= extraction_fraction_thresh and use_abstension:
        print(f"过滤后preds数量: {len(preds)}")
        print(f"过滤后golds数量: {len(golds)}")
        
        for i, (pred, gold) in enumerate(zip(preds[:10], golds[:10])):  # 只显示前10个
            print(f"过滤后样本{i}: pred={pred}, gold={gold}")
    # ===== 调试信息结束 =====
    
    for pred, gold in zip(preds, golds):
        # 标准化pred，处理各种空值情况
        pred_is_empty = False
        
        if pred is None or pred == [] or pred == [[]]:
            pred_is_empty = True
        elif isinstance(pred, list):
            # 检查[null]或[[null]]等情况
            if all(x is None or x == [] or x == '' for x in pred):
                pred_is_empty = True
            elif len(pred) == 1 and isinstance(pred[0], list):
                # 检查[[null]]或[['']]等情况
                if all(x is None or x == '' for x in pred[0]):
                    pred_is_empty = True
        elif isinstance(pred, str) and not pred.strip():
            pred_is_empty = True
            
        if pred_is_empty:
            pred_toks = []
        elif isinstance(pred, str):
            pred_toks = pred.split()
        elif isinstance(pred, list) and len(pred) == 1 and isinstance(pred[0], list):
            # 处理[[value]]格式
            pred_toks = []
            for x in pred[0]:
                if x is not None and str(x).strip():
                    pred_toks.extend(str(x).split())
        else:
            # 处理其他列表格式
            pred_toks = []
            for x in pred:
                if x is not None and str(x).strip():
                    pred_toks.extend(str(x).split())

        # 标准化gold
        if isinstance(gold, str):
            gold_toks = gold.split()
        else:
            # 处理列表格式的gold
            if isinstance(gold, list) and gold:
                if isinstance(gold[0], list):
                    # 处理[[value]]格式
                    gold_toks = []
                    for sublist in gold:
                        if isinstance(sublist, list):
                            gold_toks.extend([str(x) for x in sublist if x is not None and str(x).strip()])
                        else:
                            if sublist is not None and str(sublist).strip():
                                gold_toks.extend(str(sublist).split())
                else:
                    # 处理[value]格式
                    gold_toks = []
                    for item in gold:
                        if item is not None and str(item).strip():
                            gold_toks.extend(str(item).split())
            else:
                gold_toks = []
        
        # 计算F1分数
        if not pred_toks:  # 预测为空
            total_f1 += 0
            f1s.append(0)
            total_recall += 0
            total_prec += 0
            # ===== 添加单个样本调试 =====
            print(f"样本{total}: pred为空, F1=0")
            # ===== 调试信息结束 =====
        else:
            common = Counter(pred_toks) & Counter(gold_toks)
            num_same = sum(common.values())
            if num_same == 0:  # 没有共同元素
                total_f1 += 0
                f1s.append(0)
                total_recall += 0
                total_prec += 0
                # ===== 添加单个样本调试 =====
                print(f"样本{total}: 无共同元素, F1=0, pred_toks={pred_toks}, gold_toks={gold_toks}")
                # ===== 调试信息结束 =====                    
            else:
                precision = 1.0 * num_same / len(pred_toks)
                recall = 1.0 * num_same / len(gold_toks)
                f1 = (2 * precision * recall) / (precision + recall)
                total_f1 += f1
                total_recall += recall
                total_prec += precision
                f1s.append(f1)
                # ===== 添加单个样本调试 =====
                print(f"样本{total}: F1={f1:.4f}, P={precision:.4f}, R={recall:.4f}, pred_toks={pred_toks}, gold_toks={gold_toks}")
                # ===== 调试信息结束 =====

        total += 1

    # 总样本数为0时返回0分
    if not total:
        print("Warning: No valid samples for evaluation")
        return 0.0, 0.0
    
    # 计算平均分数    
    f1_avg = total_f1 / total
    f1_median = np.percentile(f1s, 50) if f1s else 0.0
    
    # ===== 添加最终结果调试 =====
    print(f"=== 最终计算结果 ===")
    print(f"总样本数: {total}")
    print(f"平均F1: {f1_avg:.4f}")
    print(f"中位数F1: {f1_median:.4f}")
    print(f"F1分数分布: {Counter([round(x, 2) for x in f1s])}")
    print(f"=== 调试结束 ===\n")
    # ===== 调试信息结束 =====
    
    return f1_avg, f1_median

def evaluate(
    all_extractions:list,
    gold_key:str, 
    field:str, 
    manifest_session=None, 
    overwrite_cache=False, 
    combiner_mode='mv',
    extraction_fraction_thresh=0.8,
    use_abstension=True,
    gold_extractions_file=None,  # NEW PARAMETER: Support for external gold standard files
):
    # ENHANCED VERSION: Added dynamic table.json loading and comprehensive gold standard handling
    # 动态加载table.json作为标准答案
    if gold_key == "gpt-4.1-mini":
        # 只使用传入的gold_extractions_file路径，不回退到lcr
        possible_table_paths = []
        if gold_extractions_file:
            possible_table_paths.append(gold_extractions_file)
            print(f"Using specified gold_extractions_file: {gold_extractions_file}")
        else:
            print("WARNING: No gold_extractions_file specified, will use empty gold standard")
        
        standard_answers_dict = {}
        table_json_path = None
        
        for path in possible_table_paths:
            if os.path.exists(path):
                table_json_path = path
                break
        
        if table_json_path:
            try:
                with open(table_json_path, 'r', encoding='utf-8') as f:
                    standard_answers_dict = json.load(f)
                print(f"成功加载table.json: {table_json_path}")
                print(f"总共加载了 {len(standard_answers_dict)} 个文件的标准答案")
            except Exception as e:
                print(f"加载table.json失败: {e}")
                standard_answers_dict = {}
        else:
            print("未找到table.json文件，使用空的标准答案字典")
        
        # 添加调试信息
        print(f"当前字段: {field}")
        print(f"all_extractions中的文件路径示例:")
        if gold_key in all_extractions:
            sample_paths = list(all_extractions[gold_key].keys())[:3]
            for path in sample_paths:
                print(f"  {path}")
        
        print(f"标准答案字典中的文件路径示例:")
        sample_standard_paths = list(standard_answers_dict.keys())[:3]
        for path in sample_standard_paths:
            print(f"  {path}")
        
        # 根据当前field动态提取对应的标准答案
        your_gold_answers = {}
        
        # 获取原始的gpt-4.1-mini数据以了解路径格式
        original_gold_answers = all_extractions.get(gold_key, {})
        
        for file_path in original_gold_answers.keys():
            # 尝试多种路径匹配方式
            matched_value = None
            
            # 1. 直接匹配
            if file_path in standard_answers_dict:
                matched_value = standard_answers_dict[file_path].get(field, "")
                print(f"直接匹配: {file_path} -> {matched_value}")
            
            # 2. 文件名匹配
            else:
                file_name = file_path.split('/')[-1]  # 获取文件名
                for std_path, attributes in standard_answers_dict.items():
                    if std_path.endswith(file_name):
                        matched_value = attributes.get(field, "")
                        print(f"文件名匹配: {file_path} -> {std_path} -> {matched_value}")
                        break
            
            # 3. 如果还是没找到，使用原始值
            if matched_value is None:
                matched_value = ""
                print(f"未找到匹配: {file_path} -> 使用空值")
            
            # 格式化为[[value]]的格式
            your_gold_answers[file_path] = [[matched_value]]
        
        # 替换原有的gpt-4.1-mini数据
        all_extractions[gold_key] = your_gold_answers
        print(f"替换了 {len(your_gold_answers)} 个文件的金标准")
        
    normalized_field_name = field
    for char in ["'", ":", "<", ">", '"', "_", "-", " ", "none"]:
        normalized_field_name = normalized_field_name.replace(char, "")

    key2golds = defaultdict(list)
    key2preds = defaultdict(list)
    total_tokens_prompted = 0

    # handle FM golds on D_eval
    gold_file2metadata = all_extractions[gold_key]
    cleaned_gold_metadata = {}
    for filepath, gold_metadata in gold_file2metadata.items():
        #print(f"\n处理金标准 {filepath}:")
        #print(f"原始金标准: {gold_metadata}")
        
        normalized_gold = normalize_value_type(gold_metadata, field)
        #print(f"normalize_value_type 后: {normalized_gold}")
        
        if len(normalized_gold) > 1:
            normalized_gold, num_toks = pick_a_gold_label(
                normalized_gold, 
                attribute=field, 
                manifest_session=manifest_session, 
                overwrite_cache=overwrite_cache
            )
            total_tokens_prompted += num_toks
            #print(f"pick_a_gold_label 后: {normalized_gold}")
        
        cleaned_gold = clean_comparison(normalized_gold, field)
        #print(f"clean_comparison 后: {cleaned_gold}")
        
        cleaned_gold_metadata[filepath] = cleaned_gold
    
    # handle function preds on D_eval
    # 在这里也过滤掉标准答案为空的文件
    for i, (key, file2metadata) in enumerate(all_extractions.items()):
        if key == gold_key:
            continue
        for filepath, metadata in file2metadata.items():
            gold_metadata = cleaned_gold_metadata[filepath]
            
            # 检查金标准是否为空，如果为空则跳过该文件
            gold_is_empty = False
            if isinstance(gold_metadata, str):
                gold_cleaned = gold_metadata.strip().lower()
                if (not gold_cleaned or 
                    gold_cleaned in ['', 'none', 'null', 'n/a', 'not available', 'not specified']):
                    gold_is_empty = True
            elif isinstance(gold_metadata, list):
                if not gold_metadata or all(not str(g).strip() for g in gold_metadata):
                    gold_is_empty = True
            elif gold_metadata is None:
                gold_is_empty = True
            
            # 记录并输出每个文件的处理状态
            # print(f"\nProcessing file: {filepath}")
            # print(f"Original gold metadata: {gold_metadata}")
            # print(f"Is gold empty: {gold_is_empty}")
            
            # 只有当金标准非空时才加入评估
            if not gold_is_empty:
                pred_metadata = normalize_value_type(metadata, field)
                cleaned_pred = clean_comparison(pred_metadata, field)
                # print(f"Original prediction: {metadata}")
                # print(f"Normalized prediction: {pred_metadata}")
                # print(f"Cleaned prediction: {cleaned_pred}")
                
                # 添加到评估集合
                key2golds[key].append(gold_metadata)
                key2preds[key].append(cleaned_pred)
            else:
                print("Skipping due to empty gold standard")

    # Handling abstensions
    metrics = {}
    for key, gold_list in key2golds.items():
        if not gold_list:  # 如果没有有效的样本
            print(f"Warning: No valid samples for key {key} and field {field}")
            metrics[key] = {
                "average_f1": 0.0,
                "median_f1": 0.0,
                "extraction_fraction": 0.0,
                "prior_average_f1": 0.0,
                "prior_median_f1": 0.0,
            }
            continue
            
        num_extractions = 0
        for gold_item in gold_list:
            if gold_item and not any(gold_item.lower() == wd for wd in ['none']):
                num_extractions += 1
        extraction_fraction = float(num_extractions) / float(len(gold_list))
        if combiner_mode == "top_k":
            # Don't use the extraction fraction in the naive setting for scoring
            extraction_fraction = 0.0
        
        preds = key2preds[key]
        f1, f1_med = text_f1(
            preds, gold_list, 
            extraction_fraction=extraction_fraction, 
            attribute=field,
            extraction_fraction_thresh=extraction_fraction_thresh,
            use_abstension=use_abstension,
            function_name=key,  # 传递函数名
        )
        priorf1, priorf1_med = text_f1(preds, gold_list, extraction_fraction=0.0, attribute=field,function_name=key)  # 传递函数名
        metrics[key] = {
            "average_f1": f1,
            "median_f1": f1_med,
            "extraction_fraction": extraction_fraction,
            "prior_average_f1": priorf1,
            "prior_median_f1": priorf1_med,
            
        } 

    return metrics, key2golds, total_tokens_prompted

# ...existing code...


# ...existing code...

def get_topk_scripts_per_field(
    script2metrics, 
    function_dictionary, 
    all_extractions,
    gold_key='', 
    k=3, 
    do_end_to_end=False, 
    keep_thresh = 0.1, 
    cost_thresh = 1, 
    combiner_mode='mv',
): 
    # ENHANCED VERSION: Complete rewrite with improved function selection logic
    # 获取所有函数键，排除金标准
    all_function_keys = [key for key in script2metrics.keys() if key != gold_key and "function" in key]
    
    print(f"\n=== 函数选择调试信息 ===")
    print(f"总函数数量: {len(all_function_keys)}")
    for key in all_function_keys[:5]:  # 显示前5个函数的分数
        metrics = script2metrics[key]
        #print(f"{key}: F1={metrics['average_f1']:.4f}, 提取率={metrics['extraction_fraction']:.4f}")
    
    # OLD VERSION LOGIC (completely different approach):
    # script2avg = dict(
    #     sorted(script2metrics.items(), 
    #     reverse=True, 
    #     key=lambda x: (x[1]['average_f1'], x[1]['median_f1']))
    # )
    # top_k_scripts = [k for k, v in script2avg.items() if k != gold_key] 
    # top_k_values = [
    #     max(v['average_f1'], v['median_f1']) for k, v in script2avg.items() if k != gold_key
    # ]
    
    if not all_function_keys:
        # 如果没有函数，返回最佳的非函数模型
        all_keys = [key for key in script2metrics.keys() if key != gold_key]
        if all_keys:
            best_key = max(all_keys, key=lambda x: script2metrics[x]['average_f1'])
            print(f"No functions available, using best model: {best_key}")
            return [best_key]
        return []
    
    # 按性能排序所有函数
    sorted_functions = sorted(all_function_keys, 
                             key=lambda x: script2metrics[x]['average_f1'], 
                             reverse=True)
    
    # 根据k值和质量阈值决定返回多少个函数
    if k == -1:
        # 应用质量过滤，使用0.5作为阈值
        qualified_functions = []
        threshold = 0.5
        for func in sorted_functions:
            f1_score = script2metrics[func]['average_f1']
            if f1_score >= threshold:
                qualified_functions.append(func)
        
        if qualified_functions:
            selected_functions = qualified_functions
            print(f"Using {len(selected_functions)} qualified functions (F1 >= {threshold})")
        else:
            # 如果没有合格的函数，选择前3个最好的
            selected_functions = sorted_functions[:3]
            print(f"No functions meet threshold {threshold}, using top 3 functions")
    else:
        # 返回前k个函数
        selected_functions = sorted_functions[:k]
        print(f"Using top {len(selected_functions)} functions for voting")
    
    # 显示最佳函数和选中的函数
    if sorted_functions:
        best_script = sorted_functions[0]
        best_score = script2metrics[best_script]['average_f1']
        print(f"Best script overall: {best_script}; Score: {best_score}")
        print(f"Selected functions: {selected_functions}")
    
    return selected_functions

