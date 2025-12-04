
from quest.core.nlp.match.table_matcher.table_join import pd_join_by_column, create_advanced_join_matcher
from quest.core.nlp.match.table_matcher.modular_matcher import MatchingConfig, create_comprehensive_matcher

from quest.core.nlp.match.table_matcher.table_join import pd_join_by_column_with_join_type
advanced_matcher = create_advanced_join_matcher()

def pd_fuse_join(left_table, right_table, left_on, right_on, column_type = 'TEXT', join_type='inner', matcher=advanced_matcher):

    print("左表的连接列：", left_on)
    print("右表的连接列：", right_on)

    if isinstance(left_on, list):
        left_on = left_on[0]
    if isinstance(right_on, list):
        right_on = right_on[0]
    inner_join_result = pd_join_by_column_with_join_type(
        left_table, right_table,
        left_on, right_on,
        column_type,
        join_type=join_type,
        matcher=matcher
    )
    return inner_join_result

