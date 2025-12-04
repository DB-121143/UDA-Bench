from parse import parser
from plan import planner
from optimize import optimizer 
from execute import executor
from utils import read_query_list, print_config_to_file

from schema.nba import NBAData

import os
from datetime import datetime
from tqdm import tqdm


def main(query_type="SF"):

    # location of query
    query_dir = "query/player"
    # read query
    query_dict = read_query_list(query_dir, query_type)

    # result save location, add timestamp
    result_dir = "result/player"
    result_dir = os.path.join(result_dir, query_type, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(result_dir, exist_ok=True)

    source_data = NBAData("player")
    start = False
    for query_name, query in tqdm(query_dict.items()):
        print("Executing: " + query_name)
        start = True
        parsed_query = parser(query)

        plan, invalid = planner(parsed_query, source_data)
        if invalid:
            print("Invalid query")
            continue
        assert plan is not None
        print("Plan:", plan)

        optimized_plan = optimizer(plan)    

        result = executor(optimized_plan)

        os.makedirs(os.path.join(result_dir, query_name), exist_ok=True)
        result.to_csv(os.path.join(result_dir, query_name, f"{query_name}.csv"), index=False, encoding='utf-8')

if __name__ == '__main__':
    main()