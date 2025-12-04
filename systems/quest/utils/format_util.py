def remove_duplicates(lst):
    return list(dict.fromkeys(lst))

def batch_split(input_list, batch_size):
    return [input_list[i:i+batch_size] for i in range(0, len(input_list), batch_size)]

def remove_duplicates_columns(lst):
    res = []
    for v in lst:
        flag = False
        for x in res:
            if(x.parse_full() == v.parse_full()):
                flag = True
                break
        if not flag:
            res.append(v)
    return res