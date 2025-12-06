# 通用测试脚本设计与规划

目标：在 `UDA-Bench` 内实现一套模块化、可扩展的评测脚本，对类 SQL 的文档抽取结果计算列级准确率和平均准确率，覆盖 Select/Filter、Aggregation、Join(只考虑inner join) 及多实体场景，遵循 `evaluation/doc/eval_acc_plan.md` 的数据与指标定义。

## 输入输出与目录约定
- **输入**：  
  - SQL 集合：`Query/{Dataset}/{Agg|Filter|Select|Mixed}/*.sql`。  
  - 属性元数据：`Query/{Dataset}/*_attributes.json`，含 `description`、`value_type`（int/float/str/multi-str）。  
  - 抽取结果：`evaluation/demo_acc_result/{Dataset}/.../{query_id}/result.csv`，列名与 SQL select 列一致。  
  - GT 表：`Query/{Dataset}/*.csv`。
- **输出（每条 SQL）**：写入 `evaluation/demo_acc_result/{Dataset}/{Task}/{query_dir}/{id}/acc_result/`，生成 `gold_result.csv`（duckdb 结果）、`matched_result.csv`、`matched_gold_result.csv`、`acc.json`（逐列 precision/recall/f1 及均值）。

## 核心流程
1. 加载 SQL+元数据：从 SQL 文件分割语句，关联 attributes.json 得到列的描述/类型。
2. 解析 SQL：用 sqlglot 提取表名、select 列、聚合函数、group by、join 键、停用列（ID 或 `{table}.ID`）。
3. 运行 GT：将对应数据集 CSV 注册到 duckdb，执行 SQL 得到 `gold_result`。
4. 结果读取：加载 `result.csv`，按解析出的列规范化列名与类型。
5. 行匹配：根据算子确定主键  
   - Select/Filter：ID；Join：参与 join 的 `{table}.ID`；Aggregation：group by 列；多实体：用户指定 `primary_key` 进一步拆分。  
   排序对齐，输出 `matched_result` / `matched_gold_result`。
6. 单元格评测：按列类型/算子分支  
   - 单值 int/float：数值比对（阈值或绝对相等,int要绝对相等，float给一个默认的阈值）。  
   - 单值 str：LLM 语义等价判断，使用列 `description` 作为提示。  
   - 多值 str：按 `||` 分割，LLM 判断匹配对数，得出 cell precision/recall ， 使用列 `description` 作为提示。 
   - 聚合函数SUM-MAX-AVG等列：relative_error = |x-gt|/gt；精度 = 1/(1+relative_error)， 使用列 `description` 作为提示。  。  
7. 指标汇总：对每列计算 precision/recall/f1，求宏平均；记录行数、缺失列、异常行。
8. 持久化：保存中间 CSV、`acc.json`（含每列指标、均值、评测模式、使用的主键与阈值）。

## 模块设计
- **配置与 CLI**  
  - 入口脚本 `evaluation/run_eval.py`（待建），参数：`--dataset`、`--sql-file`、`--query-id`、`--result-csv`、`--primary-key`、`--output-dir`、`--llm-provider`。  
  - 默认路径解析：基于数据集名推导 attributes.json、gt 目录、输出目录。  
  - 全局参数：数值容忍度（默认 0）、多值分隔符（`||`）、LLM 重试与缓存路径。
- **SQL 与元数据解析层**  
  - `SqlParser`: 使用 sqlglot，输出列清单（含是否聚合、聚合类型、别名）、group by、join on、涉及表。  
  - `QueryManifest`: 汇总 SQL 文本、列的 `description/value_type`、停用列，供评测与预处理共用。  
  - `SqlPreprocessor`: 读取多 SQL 文件，分割语句，提取相关属性，生成 per-query 目录与 `sql.json`。
- **GT 执行层**  
  - `GtRunner`: 在 duckdb 中注册数据集 CSV（自动用文件名作表名），执行 SQL，返回 DataFrame。  
  - 类型规范化：依据 attributes.json 将列转换为 int/float/str。
- **结果载入与规范化**  
  - `ResultLoader`: 读取 `result.csv`，补充缺失列为 `None`，校验停用列存在。  
  - 列名标准化（去反引号、大小写统一）。
- **行匹配与多实体处理**  
  - `RowMatcher`: 基于主键列进行 inner join；支持主键缺失/重复告警；可传 `primary_key` 在同一 ID 组内做二次匹配。  
  - 输出对齐后的两个 DataFrame。
- **单元格比对器**  
  - 基类 `CellComparator`；具体实现：`NumericComparator`、`StringLLMComparator`、`MultiValueComparator`、`AggComparator`。  
  - LLM 抽象：`LlmClient` 接口（支持 openai/azure/本地），内置 prompt（包含列描述、示例格式）。提供文件缓存避免重复调用。
- **指标聚合器**  
  - `MetricCalculator`：按列计算 P/R/F1；多值使用 cell precision/recall；聚合列使用 relative_error。  
  - 汇总：`macro_precision/recall/f1`、行计数、未评测列列表。
- **输出与日志**  
  - `ResultWriter`: 保存 gold/matched CSV 与 `acc.json`。  
  - 日志：标准输出 + 文件（debug 级别记录 LLM 交互开关、失败样本）。
- **测试与校验**  
  - 单元测试：SQL 解析、主键匹配、多值拆分、聚合精度计算。  
  - 集成冒烟：对少量示例 SQL 运行全链路，校验输出文件结构与指标。

## SQL 预处理（拆分与属性落盘）
- 目标：给定包含多条 SQL 的文件（如 `Query/Player/Select/select_queries_player.sql`），逐条编号拆分，生成评测所需的 `sql.json` 并创建对应目录，便于随后放置 `result.csv`。  
- 输出目录：`evaluation/demo_acc_result/{Dataset}/{Task}/{sql_file_stem}/{idx}/sql.json`，`idx` 从 1 递增；同时创建同级 `result.csv` 占位（可选，若不存在则跳过）。  
- 属性收集：解析 SQL 中涉及的表与列，结合 `*_attributes.json` 仅保留相关列，格式示例如下：
```
{
  "sql": "{SQL 语句字符串}",
  "table1": {
    "attr1": {"value_type": "int|float|str|multi-str", "description": "..."},
    "attr2": {"value_type": "...", "description": "..."}
  },
  "table2": {
    "attrx": {"value_type": "...", "description": "..."}
  }
}
```
- 逻辑：  
  1) 读取 SQL 文件并用 sqlglot 分割出完整语句；  
  2) 调用 `SqlParser` 提取涉及表/列；  
  3) 从 attributes.json 过滤对应字段，缺失列记录告警；  
  4) 生成目录与 `sql.json`，存储sql与sql中涉及到的属性信息。  
  5) 返回生成的 query 列表，供后续评测 CLI 批量运行。

## 实现计划（供审阅）
1. 搭建 `evaluation` 下的 Python 包骨架与入口 CLI，配置依赖（duckdb、sqlglot、pydantic/attrs、pandas，litellm LLM 客户端可插拔）。  
2. 完成 `SqlParser` 与 `QueryManifest`，支持 select/filter/agg/join 的关键信息抽取，同时实现 SQL 预处理器生成 per-query `sql.json`。  
3. 实现 `GtRunner`，自动注册数据集 CSV，支持别名与路径映射，输出 `gold_result.csv`。  
4. 构建 `ResultLoader` + `RowMatcher`，覆盖主键逻辑（ID/group by/join/primary_key），生成 matched 结果。  
5. 编写 `CellComparator` 系列与 `MetricCalculator`，实现单值、多值、聚合的指标计算，产出 per-column 与平均指标。  
6. 打通 `ResultWriter` 与 CLI，按约定目录落盘全部中间/最终文件。  
7. 添加单元测试与一个最小示例 SQL 的端到端冒烟脚本，验证指标与输出结构。  
8. 文档更新：在 `evaluation/doc` 补充使用说明、LLM 配置示例、可选阈值/主键参数说明。

## tips
- LLM 提供方式：默认走环境变量密钥的 OpenAI 兼容接口，如需自定义需提供 API endpoint。 参考 evaluation/ref_code/eval_on_multiEntity.py代码中 batch_completion 和环境变量的设置方法。
- 多实体场景的 `primary_key` 需由任务配置或命令行传入，如果没有传入，默认使用select后面的第1个属性列作为`primary_key`并给出log。  
- 输出文件格式沿用 demo 结构，不新增额外文件（除日志）。
