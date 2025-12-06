#!/usr/bin/env python3
"""
Aggregate results from all ablation experiments.

Usage:
    python aggregate_results.py
    python aggregate_results.py --results_dir ~/Deep_Learning/Results
    python aggregate_results.py --ablation_type learning_rate
"""

import argparse
import json
import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser(description='Aggregate ablation experiment results')
    parser.add_argument(
        '--results_dir',
        type=str,
        default='~/Deep_Learning/Results',
        help='Directory containing results (default: ~/Deep_Learning/Results)'
    )
    parser.add_argument(
        '--ablation_type',
        type=str,
        choices=['learning_rate', 'epochs', 'weight_decay', 'domain_transfer', 'all'],
        default='all',
        help='Type of ablation to aggregate (default: all)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='ablation_summary.csv',
        help='Output CSV filename (default: ablation_summary.csv)'
    )
    return parser.parse_args()

def load_all_results(results_dir):
    """Load all test_results.json files from the results directory."""
    results_dir = Path(results_dir).expanduser()
    all_results = []
    
    for subdir in results_dir.iterdir():
        if subdir.is_dir():
            results_file = subdir / 'test_results.json'
            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        result = json.load(f)
                        result['run_folder'] = subdir.name
                        all_results.append(result)
                except Exception as e:
                    print(f"Warning: Could not load {results_file}: {e}")
    
    return all_results

def extract_ablation_info(run_name):
    """
    Extract ablation type and variant from run name.
    
    Expected format: roberta_{dataset}_{ablation_info}
    Examples:
        - roberta_gazawar_lr-1e-5
        - roberta_sentiment140_epochs-10
        - roberta_gazawar_wd-0.01
        - roberta_gazawar_domain-transfer
    """
    parts = run_name.split('_')
    
    if len(parts) < 3:
        return None, None
    
    dataset = parts[1]
    ablation_info = '_'.join(parts[2:])
    
    # Determine ablation type
    if ablation_info.startswith('lr-'):
        ablation_type = 'learning_rate'
        variant = ablation_info.replace('lr-', '')
    elif ablation_info.startswith('epochs-'):
        ablation_type = 'epochs'
        variant = ablation_info.replace('epochs-', '')
    elif ablation_info.startswith('wd-'):
        ablation_type = 'weight_decay'
        variant = ablation_info.replace('wd-', '')
    elif 'domain-transfer' in ablation_info or 'from-scratch' in ablation_info:
        ablation_type = 'domain_transfer'
        variant = 'pretrain_sentiment140' if 'domain-transfer' in ablation_info else 'from_scratch'
    elif 'combined' in ablation_info:
        ablation_type = 'combined'
        variant = 'best'
    else:
        ablation_type = 'unknown'
        variant = ablation_info
    
    return ablation_type, variant

def create_comparison_plots(df, ablation_type, output_dir):
    """Create comparison plots for a specific ablation type."""
    subset = df[df['ablation_type'] == ablation_type].copy()
    
    if len(subset) == 0:
        print(f"No results found for ablation type: {ablation_type}")
        return
    
    # Sort by variant for better visualization
    if ablation_type == 'learning_rate':
        # Custom sort for learning rates
        lr_order = ['1e-5', '2e-5', '3e-5', 'scheduler']
        subset['variant'] = pd.Categorical(subset['variant'], categories=lr_order, ordered=True)
        subset = subset.sort_values('variant')
    else:
        subset = subset.sort_values('variant')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{ablation_type.replace("_", " ").title()} Ablation Results', fontsize=16)
    
    # Group by dataset for comparison
    for dataset in subset['dataset'].unique():
        dataset_subset = subset[subset['dataset'] == dataset]
        label = dataset.capitalize()
        
        # Plot accuracy
        axes[0, 0].plot(range(len(dataset_subset)), dataset_subset['accuracy'], 
                       marker='o', label=label, linewidth=2)
        
        # Plot F1 score
        axes[0, 1].plot(range(len(dataset_subset)), dataset_subset['f1_macro'], 
                       marker='s', label=label, linewidth=2)
        
        # Plot precision
        axes[1, 0].plot(range(len(dataset_subset)), dataset_subset['precision_macro'], 
                       marker='^', label=label, linewidth=2)
        
        # Plot recall
        axes[1, 1].plot(range(len(dataset_subset)), dataset_subset['recall_macro'], 
                       marker='d', label=label, linewidth=2)
    
    # Set labels and titles
    axes[0, 0].set_title('Accuracy')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xticks(range(len(subset['variant'].unique())))
    axes[0, 0].set_xticklabels(subset['variant'].unique(), rotation=45)
    
    axes[0, 1].set_title('F1 Score (Macro)')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xticks(range(len(subset['variant'].unique())))
    axes[0, 1].set_xticklabels(subset['variant'].unique(), rotation=45)
    
    axes[1, 0].set_title('Precision (Macro)')
    axes[1, 0].set_xlabel('Variant')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xticks(range(len(subset['variant'].unique())))
    axes[1, 0].set_xticklabels(subset['variant'].unique(), rotation=45)
    
    axes[1, 1].set_title('Recall (Macro)')
    axes[1, 1].set_xlabel('Variant')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xticks(range(len(subset['variant'].unique())))
    axes[1, 1].set_xticklabels(subset['variant'].unique(), rotation=45)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / f'{ablation_type}_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison plot: {output_path}")

def create_training_time_plot(df, output_dir):
    """Create a plot comparing training times across ablations."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Group by ablation type and dataset
    for ablation_type in df['ablation_type'].unique():
        if ablation_type == 'unknown':
            continue
        subset = df[df['ablation_type'] == ablation_type]
        
        for dataset in subset['dataset'].unique():
            dataset_subset = subset[subset['dataset'] == dataset]
            label = f"{ablation_type} ({dataset})"
            ax.plot(range(len(dataset_subset)), dataset_subset['training_time_minutes'],
                   marker='o', label=label, linewidth=2)
    
    ax.set_title('Training Time Comparison')
    ax.set_xlabel('Variant Index')
    ax.set_ylabel('Training Time (minutes)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'training_time_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved training time plot: {output_path}")

def print_summary_statistics(df):
    """Print summary statistics for each ablation type."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS BY ABLATION TYPE")
    print("="*80 + "\n")
    
    for ablation_type in sorted(df['ablation_type'].unique()):
        if ablation_type == 'unknown':
            continue
        
        subset = df[df['ablation_type'] == ablation_type]
        
        print(f"\n{ablation_type.upper().replace('_', ' ')}")
        print("-" * 80)
        
        for dataset in sorted(subset['dataset'].unique()):
            dataset_subset = subset[subset['dataset'] == dataset]
            
            print(f"\n  Dataset: {dataset}")
            print(f"  {'Variant':<20} {'Accuracy':<12} {'F1 (macro)':<12} {'Time (min)':<12}")
            print(f"  {'-'*56}")
            
            for _, row in dataset_subset.iterrows():
                print(f"  {row['variant']:<20} {row['accuracy']:<12.4f} {row['f1_macro']:<12.4f} {row['training_time_minutes']:<12.2f}")
            
            # Find best performer
            best_idx = dataset_subset['f1_macro'].idxmax()
            best_row = dataset_subset.loc[best_idx]
            print(f"\n  ✅ Best: {best_row['variant']} (F1={best_row['f1_macro']:.4f})")

def find_best_configurations(df):
    """Find the best configuration from each ablation type."""
    print("\n" + "="*80)
    print("BEST CONFIGURATIONS FOR COMBINED MODEL")
    print("="*80 + "\n")
    
    best_configs = {}
    
    for dataset in ['gazawar', 'sentiment140']:
        print(f"\nDataset: {dataset.upper()}")
        print("-" * 80)
        
        dataset_df = df[df['dataset'] == dataset]
        
        for ablation_type in ['learning_rate', 'epochs', 'weight_decay']:
            subset = dataset_df[dataset_df['ablation_type'] == ablation_type]
            
            if len(subset) == 0:
                continue
            
            best_idx = subset['f1_macro'].idxmax()
            best_row = subset.loc[best_idx]
            
            print(f"\n{ablation_type.replace('_', ' ').title()}:")
            print(f"  Best variant: {best_row['variant']}")
            print(f"  F1 Score: {best_row['f1_macro']:.4f}")
            print(f"  Accuracy: {best_row['accuracy']:.4f}")
            
            best_configs[f"{dataset}_{ablation_type}"] = best_row['variant']
        
        # Check domain transfer
        domain_subset = dataset_df[dataset_df['ablation_type'] == 'domain_transfer']
        if len(domain_subset) > 0 and dataset == 'gazawar':
            best_idx = domain_subset['f1_macro'].idxmax()
            best_row = domain_subset.loc[best_idx]
            
            print(f"\nDomain Transfer:")
            print(f"  Best approach: {best_row['variant']}")
            print(f"  F1 Score: {best_row['f1_macro']:.4f}")
            print(f"  Accuracy: {best_row['accuracy']:.4f}")
            
            best_configs[f"{dataset}_domain_transfer"] = best_row['variant']
    
    return best_configs

def main():
    args = parse_args()
    
    # Load all results
    print(f"Loading results from: {args.results_dir}")
    all_results = load_all_results(args.results_dir)
    
    if not all_results:
        print("No results found!")
        return
    
    print(f"Found {len(all_results)} result files")
    
    # Extract ablation info and create dataframe
    for result in all_results:
        if 'run_name' in result:
            ablation_type, variant = extract_ablation_info(result['run_name'])
            result['ablation_type'] = ablation_type
            result['variant'] = variant
        else:
            result['ablation_type'] = 'unknown'
            result['variant'] = 'unknown'
    
    df = pd.DataFrame(all_results)
    
    # Filter by ablation type if specified
    if args.ablation_type != 'all':
        df = df[df['ablation_type'] == args.ablation_type]
        print(f"Filtered to ablation type: {args.ablation_type}")
    
    if len(df) == 0:
        print("No results match the filter criteria!")
        return
    
    # Save aggregated results
    output_dir = Path(args.results_dir).expanduser()
    output_path = output_dir / args.output
    df.to_csv(output_path, index=False)
    print(f"\nSaved aggregated results to: {output_path}")
    
    # Print summary statistics
    print_summary_statistics(df)
    
    # Find best configurations
    best_configs = find_best_configurations(df)
    
    # Create comparison plots
    print("\n" + "="*80)
    print("GENERATING COMPARISON PLOTS")
    print("="*80 + "\n")
    
    if args.ablation_type == 'all':
        for ablation_type in ['learning_rate', 'epochs', 'weight_decay', 'domain_transfer']:
            create_comparison_plots(df, ablation_type, output_dir)
    else:
        create_comparison_plots(df, args.ablation_type, output_dir)
    
    # Create training time comparison
    if args.ablation_type == 'all':
        create_training_time_plot(df, output_dir)
    
    print("\n✅ Analysis complete!")
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Summary CSV: {args.output}")
    print(f"  - Comparison plots: *_comparison.png")
    
    # Suggest next steps
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Review the best configurations printed above")
    print("2. Check if 10 epochs showed significant improvement (>2% F1)")
    print("   - If yes, consider running 15 epochs ablation")
    print("3. Run combined model with best hyperparameters:")
    print("\n   Example command for GazaWar:")
    
    if 'gazawar_learning_rate' in best_configs:
        lr = best_configs['gazawar_learning_rate']
        epochs = best_configs.get('gazawar_epochs', 5)
        wd = best_configs.get('gazawar_weight_decay', 0.01)
        use_scheduler = '--use_scheduler' if lr == 'scheduler' else ''
        
        print(f"\n   python run_ablation.py --ablation_type combined --dataset gazawar \\")
        print(f"       --learning_rate {lr if lr != 'scheduler' else '2e-5'} \\")
        print(f"       --epochs {epochs} \\")
        print(f"       --weight_decay {wd} {use_scheduler}")

if __name__ == "__main__":
    main()
