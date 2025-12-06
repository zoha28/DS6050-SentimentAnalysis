#!/usr/bin/env python3
"""
run_ablation.py

Simple CLI wrapper for running ablation experiments using train_model_ablations.py.

Usage Examples:
---------------
python run_ablation.py --ablation_type learning_rate --variant 1e-5 --dataset gazawar
python run_ablation.py --ablation_type epochs --variant 10 --dataset sentiment140
python run_ablation.py --ablation_type domain_transfer --variant pretrain_sentiment140 --dataset gazawar
python run_ablation.py --ablation_type domain_transfer --variant from_scratch --dataset gazawar
"""

import argparse
import json
import sys

# Import from your training script
from train_model_ablation import (
    ABLATION_CONFIGS,
    run_ablations
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run an ablation experiment for sentiment classification."
    )

    parser.add_argument(
        "--ablation_type",
        type=str,
        required=True,
        help="Ablation category to run. Must be one of: " +
             ", ".join(list(ABLATION_CONFIGS.keys()))
    )

    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        help="Variant value for the specified ablation type."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["gazawar", "ipc", "sentiment140"],
        help="Dataset to train on."
    )

    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=None,
        help="Optional path to pretrained model (only used for some domain-transfer variants)."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    ablation_type = args.ablation_type
    variant = args.variant
    dataset_name = args.dataset
    pretrained_model_path = args.pretrained_model_path

    if ablation_type in 'weight_decay':
        variant_val = float(variant)
    elif ablation_type == 'epochs':
        variant_val = int(variant)
    else:  # domain_transfer or other string-based variants
        variant_val = variant
        
    # Validate ablation type
    if ablation_type not in ABLATION_CONFIGS:
        print(f"[ERROR] Unknown ablation type: {ablation_type}")
        print("Valid options:", ", ".join(list(ABLATION_CONFIGS.keys())))
        sys.exit(1)

    valid_variants = ABLATION_CONFIGS[ablation_type]["variants"]

    # Validate variant
    if variant_val not in valid_variants:
        print(f"[ERROR] Variant '{variant}' is not valid for ablation '{ablation_type}'.")
        print("Valid variants:", ", ".join([str(v) for v in valid_variants]))
        sys.exit(1)

    print("\n" + "=" * 100)
    print(f"Starting Ablation Run")
    print("=" * 100)
    print(f"Ablation Type : {ablation_type}")
    print(f"Variant       : {variant}")
    print(f"Dataset       : {dataset_name}")
    print(f"Pretrained    : {pretrained_model_path}")
    print("=" * 100 + "\n")

    # Run the actual ablation experiment
    model, results = run_ablations(
        ablation_type=ablation_type,
        variant=variant_val,
        dataset_name=dataset_name,
        pretrained_model_path=pretrained_model_path
    )

    # Print results for quick inspection
    if results is not None:
        print("\nExperiment Complete!")
        print(json.dumps(results, indent=2))
    else:
        print("\nExperiment finished, but results were None. Check logs for errors.")


if __name__ == "__main__":
    main()
