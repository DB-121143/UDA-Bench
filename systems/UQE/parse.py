from plan import planner

def split_query(query):
    parts = []
    start = query.find('SELECT')
    end = query.find('FROM')
    select_clause = query[start+7:end].split(',')
    select_clause = [clause.strip() for clause in select_clause]
    parts.append(select_clause)

    start = query.find('FROM')
    end = query.find('WHERE')
    if end == -1:
        end = query.find('GROUP BY')
        if end == -1:
            end = query.find('ORDER BY')
            if end == -1:
                end = query.find('LIMIT')
                if end == -1:
                    end = len(query)
    query[start+5:end].split(',')
    parts.append(query[start+5:end].split(','))

    start = query.find('WHERE')
    if start == -1:
        parts.append([])
    else:
        end = query.find('GROUP BY')
        if end == -1:
            end = query.find('ORDER BY')
            if end == -1:
                end = query.find('LIMIT')
                if end == -1:
                    end = len(query)
        parts.append(query[start+6:end])
    
    start = query.find('GROUP BY')
    if start == -1:
        parts.append([])
    else:
        end = query.find('ORDER BY')
        if end == -1:
            end = query.find('LIMIT')
            if end == -1:
                end = len(query)
        parts.append(query[start+9:end].split(','))
    
    start = query.find('ORDER BY')
    if start == -1:
        parts.append([])
    else:
        end = query.find('LIMIT')
        if end == -1:
            end = len(query)
        parts.append(query[start+9:end].split(','))
    
    start = query.find('LIMIT')
    if start == -1:
        parts.append([])
    else:
        parts.append(query[start+6:].split(','))

    return parts

def parser(query):
    query = query.rstrip(';')
    parts = split_query(query)

    return parts
