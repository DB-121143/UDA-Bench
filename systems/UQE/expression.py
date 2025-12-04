from filter_sample import sample_text_rows, sample_csv_rows, sample_image_rows, sample_merged_rows

class Expression:
    def __init__(self, name=None):
        self.name = name
        self.ok_col_type_without_llm = ['int', 'float', 'varchar']
    
    def check_col_ok_without_llm(self, col_type):
        if col_type not in self.ok_col_type_without_llm:
            return False
        else:
            return True

    def __repr__(self):
        return f'{self.name}'

class ComparisonExpr(Expression):
    def __init__(self, comp_unit):
        super().__init__('COMP')
        comp_unit = comp_unit.strip()
        # print(comp_unit)
        if comp_unit.find('<=') != -1:
            self.left, self.right = comp_unit.split('<=')
            self.op = '<='
        elif comp_unit.find('>=') != -1:
            self.left, self.right = comp_unit.split('>=')
            self.op = '>='
        elif comp_unit.find('<>') != -1:
            self.left, self.right = comp_unit.split('<>')
            self.op = '<>'
        elif comp_unit.find('>') != -1:
            self.left, self.right = comp_unit.split('>')
            self.op = '>'
        elif comp_unit.find('<') != -1:
            self.left, self.right = comp_unit.split('<')
            self.op = '<'
        elif comp_unit.find('==') != -1:
            self.left, self.right = comp_unit.split('==')
            self.op = '='
        elif comp_unit.find('=') != -1:
            self.left, self.right = comp_unit.split('=')
            self.op = '='

        self.left = self.left.strip()
        self.right = self.right.strip()
        if self.right[0] == '\'' or self.right[0] == '\"':
            self.right = self.right[1:-1]
        self.left_is_value = False
        self.right_is_value = True

    def get_row_indices_structured(self, left_type, df):
        op = self.op
        left = self.left
        right = self.right
        if self.right_is_value:
            if left_type == 'int':
                right = int(right)
            elif left_type == 'float':
                right = float(right)
        row_indices = []
        if op == '=':
            row_indices = df[df[left] == right].index.tolist()
        elif op == '<':
            row_indices = df[df[left] < right].index.tolist()
        elif op == '>':
            row_indices = df[df[left] > right].index.tolist()
        elif op == '<=':
            row_indices = df[df[left] <= right].index.tolist()
        elif op == '>=':
            row_indices = df[df[left] >= right].index.tolist()
        elif op == '<>':
            row_indices = df[df[left] != right].index.tolist()
        else:
            raise ValueError("Operator not supported")
        return row_indices
    
    def get_row_indices_unstructured(self, left_type, df, dataset_schema):
        op = self.op
        left_col = self.left.split('.')[0]
        left_attr = self.left.split('.')[1]
        right = self.right
        if left_type == 'text':
            sampled_row_index = sample_text_rows(df, 
                                                 left_col, left_attr, right, op, 
                                                 dataset_schema)
        elif left_type == 'csv':
            sampled_row_index = sample_csv_rows(df, 
                                                left_col, left_attr, right, op, 
                                                dataset_schema)
        elif left_type == 'image':
            sampled_row_index = sample_image_rows(df, left_col, left_attr, right, op, dataset_schema)

        row_indices = []
        row_indices = sampled_row_index
        return row_indices

    def __repr__(self):
        return f'{self.name}({self.left} {self.op} {self.right})'


class ConjunctionAndExpr(Expression):
    def __init__(self, expressions):
        super().__init__('AND')
        self.children = []
        for expression in expressions:
            self.add_child(expression)
    
    def add_child(self, child):
        if not isinstance(child, Expression):
            raise ValueError("Child must be an instance of Expression")
        if not isinstance(child, ComparisonExpr):
            raise ValueError("Child must be an instance of ComparisonExpr")
        self.children.append(child)
    
    def get_row_indices_unstructured(self, df, dataset_schema):
        cols_to_add = {}    # col: [[left_attr, right, op]]

        for expr in self.children:
            assert isinstance(expr, ComparisonExpr)
            left = expr.left
            left_type = dataset_schema.get_col_type(left)
            if not self.check_col_ok_without_llm(left_type):
                op = expr.op
                left_col = expr.left.split('.')[0]
                left_attr = expr.left.split('.')[1]
                right = expr.right
                if left_col not in cols_to_add:
                    cols_to_add[left_col] = []
                cols_to_add[left_col].append([left_attr, right, op])
        
        sampled_row_index = sample_merged_rows(df, cols_to_add, dataset_schema)
        return sampled_row_index

    def is_empty(self):
        return not self.children
    
    def __repr__(self) -> str:
        return f'{self.name}({self.children})'

class ConjunctionOrExpr(Expression):
    def __init__(self, expressions):
        super().__init__('OR')
        self.children = []
        for expression in expressions:
            self.add_child(expression)
    
    def add_child(self, child):
        if not isinstance(child, Expression):
            raise ValueError("Child must be an instance of Expression")
        if not isinstance(child, ConjunctionAndExpr):
            raise ValueError("Child must be an instance of ConjunctionAndExpr")
        self.children.append(child)
    
    def is_empty(self):
        return not self.children
    
    def __repr__(self):
        return f'{self.name}({self.children})'
