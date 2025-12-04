import json
import copy
import numbers

class ClassToJson(object):
    def __init__(self):
        pass

    # no set / 
    def visit_json(self, node):
        print(node, " : " ,node.__class__.__name__.lower())
        if not hasattr(node, '__dict__'):
            if isinstance(node, numbers.Number) or isinstance(node, str) or isinstance(node, bool):
                return node
            return node.__class__.__name__.lower()
        attributes = {}
        node_attrs = copy.copy(vars(node))
        for attr, value in node_attrs.items():
            if isinstance(value, set):
                lst = list(value)
                attributes[attr] = lst
            elif isinstance(value, list):
                res = []
                for v in value:
                    res.append(self.visit_json(v))
                attributes[attr] = res
            else:
                # dict or Classs or number or str or bool
                attributes[attr] = self.visit_json(value)
        return attributes

    def toJson(self, root):
        js = self.visit_json(root)
        js = json.dumps(js, indent=4)
        return js