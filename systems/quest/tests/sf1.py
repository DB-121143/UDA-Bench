import sys
sys.path.append('../../quest') # Project root path

import os
from quest.sql.parser import sqlparser
from quest.utils.class2json import ClassToJson
from quest.sql.planner.logical import LogicalPlanner
from quest.sql.planner.physical import TextPhysicalPlanner
from quest.sql.processer.processer import Processer
from quest.db.indexer.indexer import GlobalIndexer, load_all_indexer
from quest.core.llm.sampler import AttrSampler
from quest.core.llm.llm_query import TextLLMQuerier, LLMInfo
from quest.utils.log import print_log
from quest.conf.settings import RELATIVE_PROJECT_ROOT_PATH
from quest.db.indexer.indexer import GlobalIndexer, build_all_indexer

nba_player_docs_dir = "YOUR_DATASET_DIR"

def run(sql, prompt):
    print(sql)

    # build AST
    ast = sqlparser.parse_sql(sql)
    jsonConverter = ClassToJson()
    js = jsonConverter.toJson(ast)
    print("AST:\n", js)

    # build logical plan

    logicalPlanner = LogicalPlanner()
    logical = logicalPlanner.build_logical_plan(ast)
    js = jsonConverter.toJson(logical)
    print_log("Logical Plan:\n",js)

    # indexer and sampler

    # gb_indexer = build_all_indexer(
    #     doc_dirs=[nba_player_docs_dir],
    #     tables_name=["player"],
    #     types=["TextDoc"],
    #     debug_flag= False
    # )

    gb_indexer = load_all_indexer(table_to_type = {"player" : "TextDoc"})
    
    gb_sampler = AttrSampler(schema = prompt)
    gb_querier = TextLLMQuerier(prompt=prompt)

    gb_sampler.try_sample(gb_indexer.get_indexer("player")[0], prompt)
    sample_table = gb_sampler.sample_table
    # 存到log/test_sample_table.csv里 RELATIVE_PROJECT_ROOT_PATH
    #sample_table.to_csv(os.path.join(RELATIVE_PROJECT_ROOT_PATH, "quest/tests/log/test_sample_table.csv"))
    #attr_2_evidence = gb_sampler.map_attr_evidence
    #print_log(attr_2_evidence)

    # build physical plan

    physicalPlanner = TextPhysicalPlanner(gb_indexer, gb_querier, sampler=gb_sampler)
    physical = physicalPlanner.build(logical)
    #js = jsonConverter.toJson(physical)
    #print_log("Physical Plan:\n",js)

    # process

    processer = Processer()
    result = processer.process(physical)
    print_log("Result Table:\n", result)

    # LLM latency
    print("query_times : ", LLMInfo.tot_query_times)
    print("input_tokens : ", LLMInfo.tot_input_tokens)
    print("output_tokens : ", LLMInfo.tot_output_tokens)


    # make sure dir exists
    result.to_csv(os.path.join(RELATIVE_PROJECT_ROOT_PATH, "quest/tests/logs/SF1.csv"))

    return result

if __name__ == "__main__":
    sql = "SELECT birth_date, olympic_gold_medals FROM player"
    prompt = \
    """
    birth_date: birth date of the player; use format YYYY/%-m/%-d (e.g., 1984/1/30).\n \
    olympic_gold_medals: number of Olympic gold medals the player has won (e.g., 3).\n
    """
    run(sql, prompt)