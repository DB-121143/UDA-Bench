# OpenQuest-完整评测系统设计需求和规划

ref_code/ 中的比较流程只是用于参考，严禁直接复制或者复用其中的代码。

# 类SQL的非结构化文档抽取系统背景

我们的研究背景是：**让非结构化文档集合，也能像数据库的二维表一样被查询。**

具体来说，我们关注的文档集由一类文档组成——**每个文档对应一个实体**（如 1 个 NBA 球员、1 个公司财报、1 篇论文）。
文档中的关键信息就被视为实体的**属性**，因此每个文档都可以映射为“表中的一行”。所以每个文档都有1个ID，与它对应的实体唯一关联。

如果用户想要**筛选文档、提取属性、做关联或统计**，这些操作其实完全可以被写成类似 SQL 的查询。

涉及到的算子分为4大类型： Extract、Filter、Aggregation、Join
Extract：用于给定1个文档和1个属性描述，从文档中提取这个属性的值。
Filter：给定1个文档和1个条件，判断该文档是否满足这个条件。
Aggregation：给定多个文档，若干个分组条件(group by列)，获取这些分组对应的某个计数值(COUNT、SUM、AVG、MAX、MIN等)
JOIN: 分别给定2个文档及其对应的Join列，按照Join列相等把它们连接在一起。(我们这里只考虑inner_join)

我们假设如果对于文档集合的1个SQL查询中的所有属性已经通过人工逐一抽取出来并填入了二维表中，那我们就可以直接用传统的关系型数据库来对这个抽取好的二维表执行SQL，从而直接获得执行结果。(即run_sql_on_gt操作)


## 一体化评测脚本需求

我现在需要为类SQL的非结构化文档抽取系统设计一套评测系统，要能够根据ground truth表(下简称gt)、sql语句、属性描述和result表 计算出select后面所有召回属性列的列级准确率以及最后的平均准确率。

已知在gt表上用关系型数据库执行SQL得到的结果是gold_result表，我们要将它与我们的文档抽取系统(比如quest)抽取得到的结果result表进行比较，从而测出准确率。
result表和gold_result表中具有停用列，不需要去算准确率，是ID列或者`f"{table_name}.id"` 列。

已知不同类型的算子和不同的属性类型数据类型对评测的逻辑会产生不同的影响，下面我来详细讲解一下：
基本属性类型：int, float, str
扩展属性类型：多值str属性，每个值之间用`||` 分隔。
- Select-Filter on 单值属性： 由于实际的文档抽取系统比如quest必须要从文档中动态抽取属性来获取result表，而LLM抽取不一定都对，所以我们可以将result表中每一列的属性值与gold_result中对应列的属性值进行比较，算出准确率。这个计算逻辑如下：
	- 将result表和gold_result表的ID列作为主键，对result表和gold_result表进行match操作，将match上的行按ID顺序排列，分别得到行能够一一对应的matched_result和matched_gold_result
	- 对于非停用列attr，将description`[attr]`的描述加入prompt，让评测LLM逐一判断匹配上的2个cell ，即matched_result`[attr][row_idx]` 和  matched_gold_result`[attr][row_idx]` 在语义/数值上是否相同 is_same`[attr][row_idx]` ，记录count(is_same`[attr][row_idx]` == True)，下简称count_right_cell
	- result的总行数是len_result, gold_result的总行数是len_gold_result， precision`[attr]` = count_right_cell / len_result , recall`[attr]` = count_right_cell / len_gold_result， precision`[attr]` = $F1 = \frac{2 \times P \times R}{P + R}$ ，其中P = precision`[attr]` , R =  recall`[attr]` 
	- avg_precision = avg(precision`[attr]`) ， 其他几个同理
- Select-Filter on 多值属性(只能是str)
	- 与Select-Filter on 单值属性基本一样，但是对于匹配上的cell的判断逻辑和最终precision`[attr]` 、recall`[attr]`  等按行汇总的评测指标计算有变化：
		- is_same`[attr][row_idx]` 改成 cell_precision`[attr][row_idx]` 和 cell_recall`[attr][row_idx]` ， 计算方式是先将matched_result`[attr][row_idx]` 和  matched_gold_result`[attr][row_idx]` 中的多值属性按分隔符`||` 切分成2个字符串列表gold_result_multivalue`[attr][row_idx]` 和 result_multivalue`[attr][row_idx]` ，长度分别为len_gold_res_multivalue和len_res_multivalue，以gold_result_multivalue`[attr][row_idx]` 列表为准，逐一判断result_multivalue`[attr][row_idx]` 列表中是否有字符串能够语义匹配上，然后把匹配上的字符串格式设为matched_multi_value_cnt, cell_precision`[attr][row_idx]`  = matched_multi_value_cnt / len_res_multivalue, cell_recall`[attr][row_idx]`  = matched_multi_value_cnt / len_gold_res_multivalue
		- count_right_cell改成sum_cell_precision和sum_cell_recall， 分别由 sum( cell_precision`[attr][row_idx]` ) 和 sum( cell_recall`[attr][row_idx]` )得来

- Aggregation
	- 任何包含Aggregation操作的SQL，其用于匹配result表和gold_result表的主键都是group by列所对应的联合主键。
	- group by的列会作为主键列，这些列必须出现在select的列中
	- select列中的聚合函数列： MAX、SUM等数值操作必须作用于数值列，Count操作默认只有`Count(*)` 
	- 聚合操作的group by列只能是单值属性，不能是多值属性。
	- 准确率计算逻辑：单个单元格的precision=recall `先算相对误差是relative_error = |x-gt|/gt *100%，再算准确率 relative_error = 1/(1+err)` , 最后整体的p = r = f1

- Join:
	- 匹配的主键列是参与join的表的{table}.ID组成的联合主键。

- 多实体：
	- 一个文档/ID可能对应多个实体，所以最后对结果行进行匹配时，同一个ID在gold_result和result表中可能会存在多个匹配上的行，类似于Aggregation，这时我们要用户指定一个列作为primary_key去对同一组ID内的行做进一步的匹配(对于Aggregation而言，Group by列就是主键列-可自动识别，而且所有的行都属于同一个组)。

## 工程实现

parse_sql操作: 使用sqlglot这种成熟的解析库，从SQL中提取必要的信息，从而辅助行匹配或者基于主键的实体匹配。

run_sql_on_gt操作：
已知在我们的benchmark中，在gt上执行测试SQL语句获得这个查询对应的gold_result时可以直接用duckdb这种关系型数据库执行(不需要使用语义操作，因为gt中已经包含了必要的信息并且进行了清洗。)

@todo 请补充上具体的工程实现细节
@todo 请根据上述文档描述，把整个测试文档脚本的文档划分出更合理的章节，写得更清楚些。

# 输入输出格式规范

## 输入
已有sql-query的格式规范如下：
```
shejunzhi@chai03:/data2/jproject/UDA-Bench/Query$ tree  ./
./
├── Art
│   ├── Agg
│   │   └── agg_queries_Art.sql
│   ├── Art_attributes.json
│   ├── Art.csv
│   ├── Filter
│   │   └── filter_queries.sql
│   ├── Mixed
│   │   └── mixed_queries.sql
│   ├── Select
│   │   └── select_queries.sql
│   └── utils.py

```
需要从 `agg_queries_Art.sql` 中逐个抽取sql语句，从Art_attributes.json中获取每个sql中对应attr的"description"、"value_type"，然后放到对应的evaluation/demo_acc_result/Player/Select/select_queries_player/{id}/sql.json 的位置，然后同理result.csv也得放到对应的路径下。

## 输出
测试结果保存路径的格式规范如下：
```
shejunzhi@chai03:/data2/jproject/UDA-Bench/evaluation/demo_acc_result$ tree  ./
./
├── Art
└── Player
    └── Select
        └── select_queries_player
            ├── 1
            │   ├── acc_result
            │   │   ├── acc.json
            │   │   ├── gold_result.csv
            │   │   ├── matched_gold_result.csv
            │   │   └── matched_result.csv
            │   ├── result.csv
            │   └── sql.json
            └── 2

```

其中Art和Player是数据集名，不同数据集内部的文件夹结构是类似的。
以player数据集为例，其中包含的gt表格包括
Query/Player/city.csv、Query/Player/manager.csv、Query/Player/player.csv、Query/Player/team.csv


