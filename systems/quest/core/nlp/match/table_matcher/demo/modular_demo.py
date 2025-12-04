"""
模块化匹配器演示程序
展示可插拔匹配策略的使用方法
"""
import sys
sys.path.append("/jproject/DocDBRoot/DocDBpro")
sys.path.append("/jproject/DocDBRoot/DocDBpro/table_matcher")

import pandas as pd
import os
from table_matcher.modular_matcher import (
    ModularTableMatcher, MatchingConfig,
    create_simple_matcher, create_semantic_matcher, create_comprehensive_matcher
)


def demo_basic_modular_matching():
    """演示基本的模块化匹配"""
    print("=" * 80)
    print("演示1: 基本模块化匹配")
    print("=" * 80)
    
    # 创建测试数据
    ground_truth = pd.DataFrame({
        'id': ['001', '002', '003', '004'],
        'company': ['北京科技有限公司', '上海贸易公司', '深圳电子科技', '广州制造企业'],
        'product': ['人工智能软件', '进出口贸易', '智能手机', '工业设备']
    })
    
    llm_extract = pd.DataFrame({
        'id': ['001', '002', '003', '004'],
        'company': ['北京科技有限责任公司', '上海贸易有限公司', '深圳电子科技公司', '广州制造业企业'],
        'product': ['AI软件', '国际贸易', '手机', '机械设备']
    })
    
    primary_keys = 'id'
    column_types = {
        'company': 'VARCHAR',
        'product': 'VARCHAR'
    }
    
    print("Ground Truth:")
    print(ground_truth)
    print("\nLLM Extract:")
    print(llm_extract)
    
    # 使用默认配置（精确匹配 + 语义相似度）
    print("\n--- 使用默认配置（精确匹配 + 语义相似度）---")
    matcher = create_semantic_matcher()
    results = matcher.match_tables(ground_truth, llm_extract, primary_keys, column_types)
    matcher.print_results(results)
    
    return results


def demo_custom_strategies():
    """演示自定义策略配置"""
    print("\n" + "=" * 80)
    print("演示2: 自定义策略配置")
    print("=" * 80)
    
    # 创建包含相似文本的测试数据
    ground_truth = pd.DataFrame({
        'id': ['001', '002', '003'],
        'description': ['高质量的产品', '快速的服务', '专业的团队'],
        'location': ['北京市朝阳区', '上海市浦东新区', '深圳市南山区']
    })
    
    llm_extract = pd.DataFrame({
        'id': ['001', '002', '003'],
        'description': ['优质产品', '快捷服务', '专业团队'],
        'location': ['北京朝阳', '上海浦东', '深圳南山']
    })
    
    primary_keys = 'id'
    column_types = {
        'description': 'TEXT',
        'location': 'VARCHAR'
    }
    
    print("Ground Truth:")
    print(ground_truth)
    print("\nLLM Extract:")
    print(llm_extract)
    
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
            results = matcher.match_tables(ground_truth, llm_extract, primary_keys, column_types)
            print(f"总体准确率: {results['overall_precision']:.4f}")
            for col, acc in results['column_precision'].items():
                print(f"  {col}: {acc:.4f}")
        except Exception as e:
            print(f"策略配置失败: {e}")


def demo_fusion_modes():
    """演示不同的融合模式"""
    print("\n" + "=" * 80)
    print("演示3: 不同融合模式")
    print("=" * 80)
    
    # 创建有争议的测试数据
    ground_truth = pd.DataFrame({
        'id': ['001', '002'],
        'text': ['苹果公司', '微软企业']
    })
    
    llm_extract = pd.DataFrame({
        'id': ['001', '002'],
        'text': ['Apple Inc.', 'Microsoft Corp']
    })
    
    primary_keys = 'id'
    column_types = {'text': 'VARCHAR'}
    
    print("Ground Truth:")
    print(ground_truth)
    print("\nLLM Extract:")
    print(llm_extract)
    
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
            results = matcher.match_tables(ground_truth, llm_extract, primary_keys, column_types)
            
            print(f"总体准确率: {results['overall_precision']:.4f}")
            # 显示详细的匹配信息
            for col, details in results['detailed_results'].items():
                for detail in details:
                    match_details = detail.get('match_details', {})
                    if 'match_result' in match_details:
                        result = match_details['match_result']
                        print(f"  Key {detail['key']}: {result.method} -> {detail['match']} (conf: {result.confidence:.3f})")
        except Exception as e:
            print(f"融合模式 {mode} 失败: {e}")


def demo_early_stopping():
    """演示早停机制"""
    print("\n" + "=" * 80)
    print("演示4: 早停机制")
    print("=" * 80)
    
    ground_truth = pd.DataFrame({
        'id': ['001', '002'],
        'text': ['完全相同的文本', '北京科技公司']
    })
    
    llm_extract = pd.DataFrame({
        'id': ['001', '002'],
        'text': ['完全相同的文本', '北京科技有限公司']
    })
    
    primary_keys = 'id'
    column_types = {'text': 'VARCHAR'}
    
    print("Ground Truth:")
    print(ground_truth)
    print("\nLLM Extract:")
    print(llm_extract)
    
    # 测试不同的早停阈值
    early_stop_thresholds = [1.0, 0.8]
    
    for threshold in early_stop_thresholds:
        print(f"\n--- 早停阈值: {threshold} ---")
        config = (MatchingConfig()
                 .add_exact_match(priority=0)
                 .add_edit_distance(threshold=0.8, priority=1)
                 .add_semantic_similarity(threshold=0.8, priority=2)
                 .set_early_stop_threshold(threshold))
        
        matcher = ModularTableMatcher(config)
        results = matcher.match_tables(ground_truth, llm_extract, primary_keys, column_types)
        
        print(f"总体准确率: {results['overall_precision']:.4f}")
        # 查看使用了哪些策略
        for col, details in results['detailed_results'].items():
            for detail in details:
                match_details = detail.get('match_details', {})
                if 'match_result' in match_details:
                    result = match_details['match_result']
                    print(f"  Key {detail['key']}: 使用策略 {result.method}")


def demo_llm_judge_with_mock():
    """演示LLM判断（使用模拟数据，因为可能没有API key）"""
    print("\n" + "=" * 80)
    print("演示5: LLM判断策略配置")
    print("=" * 80)
    
    print("注意: 这个演示展示如何配置LLM判断策略")
    print("如果您有OpenAI API key，可以取消注释相关代码进行实际测试")
    
    ground_truth = pd.DataFrame({
        'id': ['001', '002'],
        'description': ['这是一个很好的产品', '服务质量非常优秀']
    })
    
    llm_extract = pd.DataFrame({
        'id': ['001', '002'],
        'description': ['产品质量很好', '服务很棒']
    })
    
    primary_keys = 'id'
    column_types = {'description': 'TEXT'}
    
    print("Ground Truth:")
    print(ground_truth)
    print("\nLLM Extract:")
    print(llm_extract)
    
    # 配置包含LLM判断的策略（需要API key）
    print("\n如何配置LLM判断策略:")
    print("""
    # 如果您有OpenAI API key，可以这样配置:
    config = (MatchingConfig()
             .set_openai_config("your-api-key-here")
             .add_exact_match(priority=0)
             .add_edit_distance(threshold=0.8, priority=1)
             .add_semantic_similarity(threshold=0.8, priority=2)
             .add_llm_judge(threshold=0.5, priority=3)
             .set_fusion_mode("priority"))
    
    matcher = ModularTableMatcher(config)
    """)
    
    # 使用不包含LLM的配置进行演示
    config = (MatchingConfig()
             .add_exact_match(priority=0)
             .add_edit_distance(threshold=0.6, priority=1)
             .add_semantic_similarity(threshold=0.7, priority=2))
    
    matcher = ModularTableMatcher(config)
    results = matcher.match_tables(ground_truth, llm_extract, primary_keys, column_types)
    
    print("\n使用非LLM策略的结果:")
    print(f"总体准确率: {results['overall_precision']:.4f}")


def demo_performance_comparison():
    """演示性能比较"""
    print("\n" + "=" * 80)
    print("演示6: 不同配置的性能比较")
    print("=" * 80)
    
    # 创建较大的测试数据集
    import random
    random.seed(42)
    
    n_rows = 20
    companies = ['科技公司', '贸易企业', '制造厂家', '服务机构', '金融公司']
    products = ['软件产品', '硬件设备', '咨询服务', '金融产品', '制造商品']
    
    ground_truth_data = []
    llm_extract_data = []
    
    for i in range(n_rows):
        company = f"{random.choice(['北京', '上海', '深圳', '广州'])}{random.choice(companies)}"
        product = f"{random.choice(['高级', '优质', '专业', '智能'])}{random.choice(products)}"
        
        ground_truth_data.append({
            'id': f'C{i+1:03d}',
            'company': company,
            'product': product
        })
        
        # 为LLM抽取数据添加一些变化
        company_extract = company
        product_extract = product
        
        if random.random() < 0.3:  # 30%概率修改公司名
            company_extract = company.replace('公司', '有限公司').replace('企业', '集团')
        
        if random.random() < 0.3:  # 30%概率修改产品名
            product_extract = product.replace('高级', '高端').replace('优质', '精品')
        
        llm_extract_data.append({
            'id': f'C{i+1:03d}',
            'company': company_extract,
            'product': product_extract
        })
    
    ground_truth = pd.DataFrame(ground_truth_data)
    llm_extract = pd.DataFrame(llm_extract_data)
    
    primary_keys = 'id'
    column_types = {
        'company': 'VARCHAR',
        'product': 'VARCHAR'
    }
    
    print(f"测试数据: {n_rows} 行")
    print("Ground Truth 前3行:")
    print(ground_truth.head(3))
    print("\nLLM Extract 前3行:")
    print(llm_extract.head(3))
    
    # 测试不同配置的性能
    configs = [
        ("简单编辑距离", MatchingConfig().add_edit_distance(threshold=0.8)),
        ("精确+编辑距离", MatchingConfig().add_exact_match().add_edit_distance(threshold=0.8)),
        ("精确+语义相似度", MatchingConfig().add_exact_match().add_semantic_similarity(threshold=0.8)),
        ("全策略(无LLM)", MatchingConfig().add_exact_match().add_edit_distance().add_semantic_similarity())
    ]
    
    import time
    
    for name, config in configs:
        print(f"\n--- {name} ---")
        try:
            start_time = time.time()
            matcher = ModularTableMatcher(config)
            results = matcher.match_tables(ground_truth, llm_extract, primary_keys, column_types)
            end_time = time.time()
            
            print(f"总体准确率: {results['overall_precision']:.4f}")
            print(f"处理时间: {end_time - start_time:.2f}秒")
            for col, acc in results['column_precision'].items():
                print(f"  {col}: {acc:.4f}")
        except Exception as e:
            print(f"配置 {name} 失败: {e}")


def main():
    """主演示程序"""
    print("模块化表格匹配器演示程序")
    print("展示可插拔匹配策略的功能")
    
    try:
        # 演示1: 基本功能
        demo_basic_modular_matching()
        
        # 演示2: 自定义策略
        demo_custom_strategies()
        
        # 演示3: 融合模式
        demo_fusion_modes()
        
        # 演示4: 早停机制
        demo_early_stopping()
        
        # 演示5: LLM判断配置
        demo_llm_judge_with_mock()
        
        # 演示6: 性能比较
        demo_performance_comparison()
        
        print("\n" + "=" * 80)
        print("模块化匹配器演示完成！")
        print("=" * 80)
        
    except Exception as e:
        print(f"演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


def interactive_demo():
    """交互式演示"""
    print("\n" + "=" * 80)
    print("交互式模块化匹配器演示")
    print("=" * 80)
    
    print("请选择要演示的功能:")
    print("1. 基本模块化匹配")
    print("2. 自定义策略配置")
    print("3. 不同融合模式")
    print("4. 早停机制")
    print("5. LLM判断配置")
    print("6. 性能比较")
    print("7. 运行所有演示")
    print("0. 退出")
    
    demos = {
        '1': demo_basic_modular_matching,
        '2': demo_custom_strategies,
        '3': demo_fusion_modes,
        '4': demo_early_stopping,
        '5': demo_llm_judge_with_mock,
        '6': demo_performance_comparison,
        '7': main
    }
    
    while True:
        try:
            choice = input("\n请输入选择 (0-7): ").strip()
            
            if choice == '0':
                print("退出演示程序")
                break
            elif choice in demos:
                demos[choice]()
            else:
                print("无效选择，请重新输入")
                
        except KeyboardInterrupt:
            print("\n程序被中断")
            break
        except Exception as e:
            print(f"发生错误: {e}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        interactive_demo()
    else:
        main()
