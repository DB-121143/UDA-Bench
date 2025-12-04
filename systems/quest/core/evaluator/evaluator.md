
# 需要添加的兼容性功能

- 强类型SQL查询支持
    - fixed和unfixed属性区分
    - 字符串类型的single value和multi value的区分

- 按列的准确率计算：
    - 匹配接口可以一次匹配一列的值，这样在使用LLM匹配时可以加快匹配速度。
