#!/bin/bash
# Batch Scripts for Running Multiple Ablation Experiments
# Choose the appropriate section based on your assignment

set -e  # Exit on error

# =============================================================================
# CONFIGURATION
# =============================================================================

# Activate your environment if needed (uncomment and modify)
# source ~/miniconda3/bin/activate your_env_name
# module load cuda/11.8  # If on Rivanna

# =============================================================================
# PERSON 1: Learning Rate Ablation - GazaWar
# =============================================================================

run_learning_rate_gaza() {
    echo "=========================================="
    echo "Person 1: Learning Rate Ablation - GazaWar"
    echo "=========================================="
    
    python ~/Deep_Learning/run_ablation.py --ablation_type learning_rate --variant 1e-5 --dataset gazawar
    echo "✅ Completed: lr=1e-5"
    
    python ~/Deep_Learning/run_ablation.py --ablation_type learning_rate --variant 2e-5 --dataset gazawar
    echo "✅ Completed: lr=2e-5 (baseline)"
    
    python ~/Deep_Learning/run_ablation.py --ablation_type learning_rate --variant 3e-5 --dataset gazawar
    echo "✅ Completed: lr=3e-5"
    
    python ~/Deep_Learning/run_ablation.py --ablation_type learning_rate --variant scheduler --dataset gazawar
    echo "✅ Completed: lr=scheduler"
    
    echo "=========================================="
    echo "Person 1: All experiments completed!"
    echo "=========================================="
}

# =============================================================================
# PERSON 2: Learning Rate Ablation - Sentiment140
# =============================================================================

run_learning_rate_140() {
    echo "=========================================="
    echo "Person 2: Learning Rate Ablation - Sentiment140"
    echo "=========================================="
    
    python ~/Deep_Learning/run_ablation.py --ablation_type learning_rate --variant 1e-5 --dataset sentiment140
    echo "✅ Completed: lr=1e-5"
    
    python ~/Deep_Learning/run_ablation.py --ablation_type learning_rate --variant 2e-5 --dataset sentiment140
    echo "✅ Completed: lr=2e-5 (baseline)"
    
    python ~/Deep_Learning/run_ablation.py --ablation_type learning_rate --variant 3e-5 --dataset sentiment140
    echo "✅ Completed: lr=3e-5"
    
    python ~/Deep_Learning/run_ablation.py --ablation_type learning_rate --variant scheduler --dataset sentiment140
    echo "✅ Completed: lr=scheduler"
    
    echo "=========================================="
    echo "Person 2: All experiments completed!"
    echo "=========================================="
}

# =============================================================================
# PERSON 3: Epochs Ablation - Both Datasets
# =============================================================================

run_epochs() {
    echo "=========================================="
    echo "Person 3: Epochs Ablation - Both Datasets"
    echo "=========================================="
    
    # GazaWar
    python ~/Deep_Learning/run_ablation.py --ablation_type epochs --variant 3 --dataset gazawar
    echo "✅ Completed: epochs=3 (gazawar)"
    
    python ~/Deep_Learning/run_ablation.py --ablation_type epochs --variant 5 --dataset gazawar
    echo "✅ Completed: epochs=5 (gazawar, baseline)"
    
    python ~/Deep_Learning/run_ablation.py --ablation_type epochs --variant 10 --dataset gazawar
    echo "✅ Completed: epochs=10 (gazawar)"
    
    # Sentiment140
    python ~/Deep_Learning/run_ablation.py --ablation_type epochs --variant 3 --dataset sentiment140
    echo "✅ Completed: epochs=3 (sentiment140)"
    
    python ~/Deep_Learning/run_ablation.py --ablation_type epochs --variant 5 --dataset sentiment140
    echo "✅ Completed: epochs=5 (sentiment140, baseline)"
    
    python ~/Deep_Learning/run_ablation.py --ablation_type epochs --variant 10 --dataset sentiment140
    echo "✅ Completed: epochs=10 (sentiment140)"
    
    echo "=========================================="
    echo "Person 3: Base experiments completed!"
    echo "=========================================="
    echo ""
    echo "📊 Now check if 10 epochs showed >2% F1 improvement"
    echo "   If yes, run the conditional 15 epochs experiments:"
    echo ""
    echo "   python run_ablation.py --ablation_type epochs --variant 15 --dataset gazawar"
    echo "   python run_ablation.py --ablation_type epochs --variant 15 --dataset sentiment140"
    echo ""
}

run_15epochs_conditional() {
    echo "=========================================="
    echo "Person 3: Conditional 15 Epochs Experiments"
    echo "=========================================="
    
    python ~/Deep_Learning/run_ablation.py --ablation_type epochs --variant 15 --dataset gazawar
    echo "✅ Completed: epochs=15 (gazawar)"
    
    python ~/Deep_Learning/run_ablation.py --ablation_type epochs --variant 15 --dataset sentiment140
    echo "✅ Completed: epochs=15 (sentiment140)"
    
    echo "=========================================="
    echo "Person 3: ALL experiments completed!"
    echo "=========================================="
}

# =============================================================================
# PERSON 4: Weight Decay Ablation - Both Datasets
# =============================================================================

run_weight_decay() {
    echo "=========================================="
    echo "Person 4: Weight Decay Ablation - Both Datasets"
    echo "=========================================="
    
    # GazaWar
    python ~/Deep_Learning/run_ablation.py --ablation_type weight_decay --variant 0.001 --dataset gazawar
    echo "✅ Completed: wd=0.001 (gazawar)"
    
    python ~/Deep_Learning/run_ablation.py --ablation_type weight_decay --variant 0.01 --dataset gazawar
    echo "✅ Completed: wd=0.01 (gazawar, baseline)"
    
    python ~/Deep_Learning/run_ablation.py --ablation_type weight_decay --variant 0.1 --dataset gazawar
    echo "✅ Completed: wd=0.1 (gazawar)"
    
    # Sentiment140
    python ~/Deep_Learning/run_ablation.py --ablation_type weight_decay --variant 0.001 --dataset sentiment140
    echo "✅ Completed: wd=0.001 (sentiment140)"
    
    python ~/Deep_Learning/run_ablation.py --ablation_type weight_decay --variant 0.01 --dataset sentiment140
    echo "✅ Completed: wd=0.01 (sentiment140, baseline)"
    
    python ~/Deep_Learning/run_ablation.py --ablation_type weight_decay --variant 0.1 --dataset sentiment140
    echo "✅ Completed: wd=0.1 (sentiment140)"
    
    echo "=========================================="
    echo "Person 4: All experiments completed!"
    echo "=========================================="
}

# =============================================================================
# PERSON 5: Domain Transfer Ablation
# =============================================================================

run_domain_transfer() {
    echo "=========================================="
    echo "Person 5: Domain Transfer Ablation"
    echo "=========================================="
    
    python ~/Deep_Learning/run_ablation.py --ablation_type domain_transfer --variant from_scratch --dataset gazawar
    echo "✅ Completed: from_scratch (baseline)"
    
    python ~/Deep_Learning/run_ablation.py --ablation_type domain_transfer --variant pretrain_sentiment140 --dataset gazawar
    echo "✅ Completed: pretrain_sentiment140 (two-stage)"
    
    echo "=========================================="
    echo "Person 5: All experiments completed!"
    echo "=========================================="
}

run_all() {
    echo "====================================================="
    echo "Running ALL ablation experiments (LR, epochs, WD, DT)"
    echo "====================================================="

    run_learning_rate_gaza
    run_epochs
    run_weight_decay
    run_domain_transfer
    run_learning_rate_140

    echo "====================================================="
    echo "All experiments completed!"
    echo "====================================================="
}

# =============================================================================
# USAGE
# =============================================================================

show_usage() {
    echo "Usage: $0 {1|2|3|4|5|all}"
    echo ""
    echo "Experiment Groups (aligned with script functions):"
    echo ""
    echo "  1   Run Person 2 experiments (Sentiment140 — learning rate ablations)"
    echo "  2   Run Person 1 experiments (GazaWar — learning rate ablations)"
    echo "  3   Run Person 3 experiments (Epoch ablations — both datasets)"
    echo "  4   Run Person 4 experiments (Weight decay ablations — both datasets)"
    echo "  5   Run Person 5 experiments (Domain transfer ablation — GazaWar)"
    echo "  all Run ALL experiments from Persons 1–5"
    echo ""
    echo "Examples:"
    echo "  $0 1"
    echo "  $0 all"
}

# =============================================================================
# MAIN
# =============================================================================

if [ $# -eq 0 ]; then
    show_usage
    exit 1
fi

case "$1" in
    1)
        run_learning_rate_140
        ;;
    2)
        run_learning_rate_gaza
        ;;
    3)
        run_epochs
        ;;
    3_conditional)
        run_15epochs_conditional
        ;;
    4)
        run_weight_decay
        ;;
    5)
        run_domain_transfer
        ;;
    all)
        run_all
        ;;
    *)
        echo "Error: Unknown option '$1'"
        echo ""
        show_usage
        exit 1
        ;;
esac

echo ""
echo "🎉 All assigned experiments completed successfully!"
echo ""
echo "Next steps:"
echo "1. Check results in ~/Deep_Learning/Results/"
echo "2. View W&B dashboard: https://wandb.ai/your-team/ipc-sentiment-ablation"
echo "3. When all team members finish, run: python aggregate_results.py"
