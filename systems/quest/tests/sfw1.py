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
from quest.sql.optimizer.optimizer_filter import OptimizerFilter

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

    print("sample now! - - - - - ")

    gb_sampler.try_sample(gb_indexer.get_indexer("player")[0], prompt)

    print("optimize now! - - - - -")

    optimizer = OptimizerFilter(indexer = gb_indexer, querier = gb_querier, sampler = gb_sampler, batch_size=30)
    opt = optimizer.build(logical)

    print("physical now! - - - - -")

    physicalPlanner = TextPhysicalPlanner(gb_indexer, gb_querier, sampler=gb_sampler)
    physical = physicalPlanner.build(opt)
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
    result.to_csv(os.path.join(RELATIVE_PROJECT_ROOT_PATH, "quest/tests/log/SFW1.csv"))

    return result

if __name__ == "__main__":

    sql = "SELECT nba_championships FROM player WHERE ((((team=='Boston Celtics' OR college=='Wake Forest University') AND nationality=='Argentina') OR position=='Backcourt') OR fiba_world_cup==0)"

    prompt = \
    """
    fiba_world_cup : number of FIBA World Cup titles the player has won (e.g., 1). \n \
    team: the current NBA team the player belongs to, or the last NBA team the player joined if not currently active (e.g., Los Angeles Lakers). \n \
    nba_championships: number of NBA championships the player has won (e.g., 4).\n \
    college: the college the player attended, if applicable (e.g., Duke University).\n \
    nationality: nationality of the player (e.g., USA, France).\n \
    position: player's primary position on the team; choose one from ['Frontcourt', 'Backcourt'].\n
    """
    
    run(sql, prompt)