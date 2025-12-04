
# 表格实体匹配python工具包

## 同表带主键逐行匹配

接口：

```通过工厂函数创建匹配器
    # 使用默认配置（精确匹配 + 语义相似度）
    print("\n--- 使用默认配置（精确匹配 + 语义相似度）---")
    matcher = create_semantic_matcher()
    results = matcher.match_tables(ground_truth, llm_extract, primary_keys, column_types)
    matcher.print_results(results)
```

```通过手动添加配置文件创建：
    # 测试不同的策略组合
    strategies_to_test = [
        ("只使用编辑距离", MatchingConfig().add_edit_distance(threshold=0.6)),
        ("精确匹配 + 编辑距离", MatchingConfig().add_exact_match().add_edit_distance(threshold=0.6)),
        ("编辑距离 + 语义相似度", MatchingConfig().add_edit_distance(threshold=0.6).add_semantic_similarity(threshold=0.7)),
    ]
    
    for name, config in strategies_to_test:
        print(f"\n--- {name} ---")
        try:
            matcher = ModularTableMatcher(config)
```

```通过设置不同匹配策略的优先级创建：
    # 测试不同融合模式
    fusion_modes = ["priority", "voting", "weighted"]
    
    for mode in fusion_modes:
        print(f"\n--- 融合模式: {mode} ---")
        try:
            config = (MatchingConfig()
                     .add_exact_match(priority=0)
                     .add_edit_distance(threshold=0.3, priority=1)  # 设置较低阈值
                     .add_semantic_similarity(threshold=0.6, priority=2)  # 设置较低阈值
                     .set_fusion_mode(mode))
            
            matcher = ModularTableMatcher(config)
```

## 2表join匹配

请使用下面的匹配器设置。
```
def create_advanced_join_matcher() -> ModularTableMatcher:
    """创建针对表连接优化的匹配器"""
    config = (MatchingConfig()
             .add_exact_match(priority=0)
             .add_edit_distance(threshold=0.80, priority=1)  # 较低的阈值以适应更多的变体
             .add_semantic_similarity(threshold=0.924, priority=2)
             .set_fusion_mode("priority"))
    return ModularTableMatcher(config)
```

