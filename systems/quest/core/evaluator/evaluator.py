
import pandas as pd
import duckdb
import re
import sqlglot

"""
SELECT marriage, AVG(age) FROM wikiart GROUP BY marriage

SELECT zodiac, MAX(age) FROM wikiart GROUP BY zodiac

SELECT century, MAX(age) FROM wikiart GROUP BY century

SELECT birth_continent, COUNT(name) FROM wikiart GROUP BY birth_continent
"""

class ExtractionEvaluator:
    def get_result_gt(self, sql,  gt_table) -> pd.DataFrame:
        # 在内存中注册gt_table为duckdb的表
        con = duckdb.connect(database=':memory:')
        con.register('wikiart', gt_table)
        # 执行SQL
        result_df = con.execute(sql).fetchdf()
        con.close()
        return result_df
    

    def align_pred_gt(self, pred_df, gt_df, primary_keys):
        # 先把primary_keys列-联合主键，加一列，然后pred_df和gt_df分别对主键列去重(如果有重复的需要给出Warning并且输出重复行的内容到stderr)，然后pred_df和gt_df分别按主键列排序之后分别返回        
        # 先生成联合主键列名
        key_col = '__joint_key__'
        # 生成联合主键列
        pred_df[key_col] = pred_df[primary_keys].astype(str).agg('||'.join, axis=1)
        gt_df[key_col] = gt_df[primary_keys].astype(str).agg('||'.join, axis=1)

        # 检查并输出重复主键
        pred_dups = pred_df[pred_df.duplicated(key_col, keep=False)]
        gt_dups = gt_df[gt_df.duplicated(key_col, keep=False)]

        if not pred_dups.empty:
            print(f"[Warning] pred_df 存在重复主键行：", file=sys.stderr)
            print(pred_dups, file=sys.stderr)
        if not gt_dups.empty:
            print(f"[Warning] gt_df 存在重复主键行：", file=sys.stderr)
            print(gt_dups, file=sys.stderr)

        # 去重
        pred_df = pred_df.drop_duplicates(key_col, keep='first')
        gt_df = gt_df.drop_duplicates(key_col, keep='first')

        # 按主键排序
        aligned_pred_df = pred_df.sort_values(by=key_col).reset_index(drop=True)
        aligned_gt_df = gt_df.sort_values(by=key_col).reset_index(drop=True)

        return aligned_pred_df, aligned_gt_df # 在debug状态下，这2个表要存下来，由于按主键排序，方便逐行对照查错。

    def cal_intersection_and_acc(self, aligned_pred_df, aligned_gt_df, primary_keys):
        pass

    def parse_and_convert_sql(self, sql: str, primary_keys = []):
        """
        尝试将sql转换为适用于duckdb的sql，用sqlglot
        并从中提取出group by的属性列名(之后join pred_table和gt_table时将被用作join列/主键列)
        并将表名、属性名、别名全部转换为小写
        并将所有表别名和列别名都替换成原始的表名和列名
        返回: (duckdb_sql: str, group_by_columns: list[str])
        """
        sql_clean = ' '.join(sql.strip().split())

        try:
            parsed = sqlglot.parse_one(sql_clean)
        except Exception as e:
            raise ValueError(f"SQL 解析失败: {e}")

        # 替换表别名和列别名为原始表名和列名
        def remove_aliases(node):
            # 替换表别名为原始表名
            if node.__class__.__name__ == "TableAlias":
                return node.this  # 返回原始表节点
            # 替换列别名为原始表达式
            if node.__class__.__name__ == "Alias":
                return node.this  # 返回原始表达式
            return node

        # 将别名替换为原始表名
        parsed = parsed.transform(remove_aliases)

        # 转小写
        for node in parsed.walk():
            # 列名
            if node.__class__.__name__ == "Column":
                if node.args.get("this"):
                    node.set("this", node.args["this"].copy().transform(lambda n: n.lower() if isinstance(n, str) else n))
                if node.args.get("table"):
                    node.set("table", node.args["table"].copy().transform(lambda n: n.lower() if isinstance(n, str) else n))
            # 表名
            if node.__class__.__name__ == "Table":
                if node.args.get("this"):
                    node.set("this", node.args["this"].copy().transform(lambda n: n.lower() if isinstance(n, str) else n))
            # 别名
            if node.args.get("alias"):
                alias = node.args["alias"]
                if hasattr(alias, "set"):
                    if alias.args.get("this"):
                        alias.set("this", alias.args["this"].copy().transform(lambda n: n.lower() if isinstance(n, str) else n))

        duckdb_sql = parsed.to_sql(dialect="duckdb")

# ...existing code...
        candidate_primary_keys = [] # group by 的列名
        group = parsed.args.get("group")
        if group:
            for g in group.expressions:
                # 处理带表名前缀的列
                if hasattr(g, "args"):
                    table = None
                    col = None
                    # 兼容 Column 节点
                    if "table" in g.args and g.args["table"]:
                        table = str(g.args["table"]).lower()
                    if "this" in g.args and hasattr(g.args["this"], "name"):
                        col = g.args["this"].name.lower()
                    elif hasattr(g, "name"):
                        col = g.name.lower()
                    if table and col:
                        candidate_primary_keys.append(f"{table}.{col}")
                    elif col:
                        candidate_primary_keys.append(col)
                    else:
                        candidate_primary_keys.append(str(g).lower())
                else:
                    candidate_primary_keys.append(str(g).lower())

        else:
            candidate_primary_keys = primary_keys
        return duckdb_sql, candidate_primary_keys



    def clean_table(self, df: pd.DataFrame, primary_keys: list[str] = []) -> pd.DataFrame:
        # 执行默认的清洗流程
        # 对于字符串类型的值，trim首尾空格，相邻token间的连续空白符统一转换成1个空格
        # 对于primary key, 统一按照字符串类型处理，如果是int，则需要转化成字符串。
        # 对于多值字符串属性，我们不需要知道它是不是多值属性，我们只需要在处理字符串时先按照分隔符'||' split，然后对split出来的每个组分别做清洗，最后再用||拼接各个组即可。如果本来是单值属性，就不会拼接上||，所以逻辑也是对的。        
        
        def clean_str(val):
            
            if pd.isnull(val):
                return val
            # 转为字符串
            val = str(val)
            # 多值处理
            parts = val.split('||')
            cleaned_parts = []
            for part in parts:
                # 去除首尾空白，合并中间连续空白
                cleaned = re.sub(r'\s+', ' ', part.strip())
                cleaned_parts.append(cleaned)
            return '||'.join(cleaned_parts)

        # 处理主键列
        for col in primary_keys:
            if col in df.columns:
                df[col] = df[col].astype(str).apply(clean_str)

        # 处理所有object类型的列（字符串列）
        for col in df.select_dtypes(include=['object']).columns:
            if col not in primary_keys:
                df[col] = df[col].apply(clean_str)

        # 列名全部转为小写，并处理table_name.attr为table_name.attr的小写
        new_columns = []
        for col in df.columns:
            if '.' in col:
                parts = col.split('.')
                parts = [p.lower() for p in parts]
                new_columns.append('.'.join(parts))
            else:
                new_columns.append(col.lower())
        df.columns = new_columns

        return df


