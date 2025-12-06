# LLM 语义匹配点与 Prompt 模板

本文枚举当前/计划中的评测脚本里会用到 LLM 比对的环节，并给出可直接落到 litellm 的 prompt 模板（system/user 双消息）。模板均假设 `temperature=0`、`max_tokens` 较小、使用 `litellm.completion` 或 `batch_completion`。

## 场景与脚本清单
- `evaluation/run_eval.py`（主入口，当前未接 LLM）：适合在 `StringLLMComparator`、`MultiValueComparator`、行对齐的二级主键兜底时调用下述模板。
- `evaluation/ref_code/eval_on_multiEntity.py`：多实体场景，列语义匹配（多值计数）与 primary_key 语义对齐均用 LLM。
- `evaluation/ref_code/eval_on_singleEntity_no_agg.py`：单实体无聚合，非主键列的语义匹配（含多值）用 LLM。
- `evaluation/ref_code/eval_on_singleEntity_with_agg.py`：单实体带聚合，非主键列语义匹配 + GROUP BY key 语义对齐用 LLM。

## Prompt 模板

### 1) 列语义等价（单值 str）
用途：`StringLLMComparator` 判断预测值与 GT 是否语义等价，输出 `YES`/`NO`。

```json
[
  {
    "role": "system",
    "content": "You judge whether a predicted cell matches the ground truth for one column. Respond with ONLY YES or NO."
  },
  {
    "role": "user",
    "content": "Column: {column_name}\nDescription: {column_description_or_None}\n\nPredicted: {pred_value}\nGround truth: {gold_value}\n\nRules:\n- Case/punctuation/spacing differences do not matter.\n- Treat synonyms, abbreviations, and obvious typos as equivalent if meaning is the same.\n- Numbers match if equal up to formatting/rounding (e.g., 1, 1.0, 01).\n- Empty/unknown is NOT equal to a specific value.\n\nAnswer YES or NO."
  }
]
```

### 2) 多值列匹配计数（multi-str，用 `batch_completion`）
用途：`MultiValueComparator` / 多实体列匹配，统计 `List A` 中有多少项能在 `List B` 找到语义等价项，返回整数。

```json
[
  {
    "role": "system",
    "content": "You count how many predicted terms have a semantic match in the ground truth list. Return only the integer."
  },
  {
    "role": "user",
    "content": "Use the column context to match terms.\nColumn: {column_name}\nDescription: {column_description_or_None}\n\nPredicted terms (List A): {pred_terms_split_by_sep}\nGround truth terms (List B): {gold_terms_split_by_sep}\n\nRules:\n- Terms are separated by '||'; treat each as an independent item.\n- A term matches if it is a synonym/paraphrase/abbreviation or a number within 1% relative error.\n- Count how many terms in List A have at least one semantic match in List B.\n\nReturn ONLY the count as an integer."
  }
]
```

注意，如果1个cell中的多值属性匹配数量超过了len(pred_list)或者len(gt_list)， 那就要取 count = min(count, len(pred_list), len(gt_list))

### 3) 主键/二级键语义对齐（多实体 primary_key）
用途：当精确主键不等时，用于判断 GT 行与预测行是否指向同一实体，返回 `YES`/`NO`。

```json
[
  {
    "role": "system",
    "content": "You decide if two mentions refer to the SAME entity. Respond ONLY YES or NO."
  },
  {
    "role": "user",
    "content": "Mention A (ground truth): {gt_value}\nMention B (prediction): {pred_value}\n\nConsider abbreviations, typos, synonyms, translations. If they clearly mean the same entity, answer YES; otherwise NO."
  }
]
```

### 4) GROUP BY / 聚合键语义对齐
用途：单实体带聚合脚本中，用于判断两个 group-by key 描述是否代表同一分组，返回 `YES`/`NO`。

```json
[
  {
    "role": "system",
    "content": "You judge whether two group-by keys describe the same bucket in an aggregated SQL result. Respond ONLY YES or NO."
  },
  {
    "role": "user",
    "content": "Columns (with descriptions):\n{key_context_lines_or_default}\n\nKey A (ground truth):\n{key_desc_gt}\n\nKey B (prediction):\n{key_desc_pred}\n\nGuidelines:\n- Each key is a set of column=value pairs; all pairs must match semantically.\n- Allow synonyms/abbreviations/wording or casing differences if meaning is unchanged.\n- Numeric values match if equal up to formatting/rounding.\n- Missing/blank/unknown is NOT equal to a specific value.\n- If any column conflicts, answer NO.\n\nReturn ONLY YES or NO."
  }
]
```

## 脚本对应的使用建议
- **run_eval.py**：目前 `StringLLMComparator`/`MultiValueComparator` 仅做规则匹配，可按上方模板 1/2 接 litellm（`settings.llm_provider != "none"` 时启用）；若未来支持二级主键语义对齐，可套用模板 3。

ref_code/ 中的比较流程只是用于参考，严禁直接复制或者复用其中的代码。

以上模板均可直接嵌入 litellm 调用，保持 `temperature=0`、小 `max_tokens`，并在 `settings.llm_model` 透传模型名以便在 `run_eval.py` 中做可插拔配置。当前文档供审阅后决定是否把 LLM 配置接入主流程。 
