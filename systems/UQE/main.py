import argparse
import sys

# 导入各个模块的main函数
from main_Art import main as main_art
from main_Art_image import main as main_art_image
from main_disease import main as main_disease
from main_drug import main as main_drug
from main_fin import main as main_fin
from main_institutes import main as main_institutes
from main_LCR import main as main_lcr
from main_player import main as main_player

# 导入配置
import config_uqe


class UQEMain:
    
    def __init__(self):
        self.dataset_handlers = {
            'art': main_art,
            'art_image': main_art_image,
            'disease': main_disease,
            'drug': main_drug,
            'finance': main_fin,
            'institutes': main_institutes,
            'lcr': main_lcr,
            'player': main_player
        }
    
    def print_config(self):
        """Print current configuration"""
        print("=" * 50)
        print("UQE Configuration:")
        print("=" * 50)
        print(f"USE_BART: {config_uqe.USE_BART}")
        print(f"BATCH_SIZE: {config_uqe.BATCH_SIZE}")
        print(f"BUDGET: {config_uqe.BUDGET}")
        print(f"AGGR_STRATEGY: {config_uqe.AGGR_STRATEGY}")
        print(f"N_CENTROIDS: {config_uqe.N_CENTROIDS}")
        print(f"N_ITER: {config_uqe.N_ITER}")
        print(f"GROUP_EXTRACT_SAMPLE_RATIO: {config_uqe.GROUP_EXTRACT_SAMPLE_RATIO}")
        print(f"AGGR_CLUSTER_SAMPLE_RATIO: {config_uqe.AGGR_CLUSTER_SAMPLE_RATIO}")
        print("=" * 50)
    
    def run_single_dataset(self, dataset_name: str, query_type: str = None):
        """Run single dataset"""
        if dataset_name not in self.dataset_handlers:
            print(f"Error: Dataset '{dataset_name}' not found!")
            return False

        print(f"Running dataset: {dataset_name}")
        
        try:
            
            # Execute corresponding main function
            self.dataset_handlers[dataset_name](query_type)
            
            print(f"Successfully completed dataset: {dataset_name}")
            return True
            
        except Exception as e:
            print(f"Error running dataset {dataset_name}: {str(e)}")
            return False
    
    def run_all_datasets(self, query_type: str = None):
        """Run all datasets"""
        print("Running all datasets...")
        self.print_config()
        
        results = {}
        for dataset_name in self.dataset_handlers.keys():
            print(f"\n{'='*20} Processing {dataset_name} {'='*20}")
            success = self.run_single_dataset(dataset_name, query_type)
            results[dataset_name] = success
        
        # Print execution summary
        print("\n" + "="*50)
        print("Execution Summary:")
        print("="*50)
        for dataset, success in results.items():
            status = "✓ SUCCESS" if success else "✗ FAILED"
            print(f"{dataset}: {status}")
        
        successful = sum(results.values())
        total = len(results)
        print(f"\nTotal: {successful}/{total} datasets completed successfully")
        
        return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='UQE - Unified Query Execution')
    parser.add_argument('--dataset', '-d', type=str, 
                       help='Dataset to run (art, art_image, disease, drug, finance, institutes, lcr, player)')
    parser.add_argument('--all', '-a', action='store_true',
                       help='Run all datasets')
    parser.add_argument('--query-type', '-q', type=str,
                       help='Override query type for all datasets')
    
    args = parser.parse_args()
    
    uqe = UQEMain()

    uqe.print_config()
    
    # 运行单个数据集
    if args.dataset:
        success = uqe.run_single_dataset(args.dataset, args.query_type)
        sys.exit(0 if success else 1)
    
    # 运行所有数据集
    if args.all:
        results = uqe.run_all_datasets(args.query_type)
        all_success = all(results.values())
        sys.exit(0 if all_success else 1)
    
    # If no arguments specified, show help
    if not any([args.dataset, args.all]):
        parser.print_help()
        print("\nExamples:")
        print("  python main.py --dataset art             # Run art dataset")
        print("  python main.py --dataset finance --query-type SFW  # Run finance with SFW queries")
        print("  python main.py --all                     # Run all datasets")


if __name__ == '__main__':
    main() 