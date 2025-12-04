from expression import Expression
from expression import ComparisonExpr, ConjunctionAndExpr, ConjunctionOrExpr

from oper import Operator
from oper import FilterOperator, ProjectOperator, GroupbyOperator, OrderbyOperator, LimitOperator, ScanOperator

def planner(parsed_query, source_data):
    invalid_query = False
    project = parsed_query[0]
    from_ = parsed_query[1]
    where = parsed_query[2]
    group_by = parsed_query[3]
    order_by = parsed_query[4]
    limit = parsed_query[5]

    oper_ = Operator()

    if from_:
        scan_oper = ScanOperator(source_data, from_)
        oper_ = scan_oper

    if where:
        or_list = where.split('OR')
        if len(or_list) > 1:
            and_expr_list = []
            for and_list in or_list:
                and_list = and_list.strip()
                if and_list.startswith('(') and and_list.endswith(')'):
                    and_list = and_list[1:-1]
                and_list = and_list.split('AND')
                comp_expr_list = []
                for and_unit in and_list:
                    and_unit = and_unit.strip()
                    and_unit = and_unit.lstrip('(')
                    and_unit = and_unit.rstrip(')')
                    comp_expr_list.append(ComparisonExpr(and_unit))
                and_expr = ConjunctionAndExpr(comp_expr_list)
                and_expr_list.append(and_expr)
            or_expr = ConjunctionOrExpr(and_expr_list)
            filter_oper_ = FilterOperator(source_data, or_expr)
        else:
            and_list = or_list[0].strip()
            if and_list.startswith('(') and and_list.endswith(')'):
                and_list = and_list[1:-1]
            and_list = and_list.split('AND')
            comp_expr_list = []
            for and_unit in and_list:
                and_unit = and_unit.strip()
                and_unit = and_unit.lstrip('(')
                and_unit = and_unit.rstrip(')')
                comp_expr_list.append(ComparisonExpr(and_unit))
            and_expr = ConjunctionAndExpr(comp_expr_list)
            filter_oper_ = FilterOperator(source_data, and_expr)
        filter_oper_.add_child(oper_)
        oper_ = filter_oper_
    
    aggr_col_list = []
    has_aggregate = False
    for col in project:
        if col.find('sum(') != -1 or col.find('avg(') != -1 or col.find('count(') != -1 or col.find('max(') != -1 or col.find('min(') != -1:
            aggr_func = col.split('(')[0].strip()
            aggr_col = col.split('(')[1].split(')')[0].strip()

            aggr_col_list.append({aggr_col: aggr_func})
            has_aggregate = True
            if col.find('max(') != -1 or col.find('min(') != -1:
                invalid_query = True
                return None, invalid_query
    if has_aggregate:
        if group_by:
            group_by_oper = GroupbyOperator(source_data, group_by, aggr_col_list)
        else:
            group_by_oper = GroupbyOperator(source_data, None, aggr_col_list)
        group_by_oper.add_child(oper_)
        oper_ = group_by_oper
    else:
        if group_by:
            raise ValueError("Group by without aggregate function")
        
    if not has_aggregate and not group_by:
        project.insert(0, 'id')
    project_oper = ProjectOperator(source_data, project)
    project_oper.add_child(oper_)
    oper_ = project_oper

    if order_by:
        order_by_oper = OrderbyOperator(source_data, order_by)
        order_by_oper.add_child(oper_)
        oper_ = order_by_oper

    if limit:
        limit_oper = LimitOperator(source_data, limit)
        limit_oper.add_child(oper_)
        oper_ = limit_oper

    oper_.print_plan()
    return oper_, invalid_query
