#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.evaluation_metrics import TransactionEvaluator
import torch

def test_evaluation_metrics():
    """Test the evaluation metrics functionality."""
    print("🧪 Testing Evaluation Metrics...")
    
    # Create sample transactions
    sample_transactions = [
        {
            "transactionAmount": {"amount": "100.50"},
            "bookingDateTime": "2024-01-15T10:30:00",
            "category": "Shopping",
            "creditorName": "Amazon"
        },
        {
            "transactionAmount": {"amount": "-50.25"},
            "bookingDateTime": "2024-01-16T14:20:00",
            "category": "Dining",
            "creditorName": "Restaurant ABC"
        },
        {
            "transactionAmount": {"amount": "2000.00"},
            "bookingDateTime": "2024-01-01T09:00:00",
            "category": "Salary",
            "creditorName": "Company XYZ"
        }
    ]
    
    # Initialize evaluator
    device = torch.device('cpu')
    evaluator = TransactionEvaluator(device=device)
    
    print("✅ Evaluator initialized successfully")
    
    # Test diversity score
    diversity_score = evaluator.calculate_diversity_score(sample_transactions)
    print(f"🎲 Diversity Score: {diversity_score:.4f}")
    
    # Test realism score
    realism_score = evaluator.calculate_realism_score(sample_transactions)
    print(f"🎭 Realism Score: {realism_score:.4f}")
    
    # Test inception score (without training)
    inception_score = evaluator.calculate_inception_score(sample_transactions)
    print(f"🏆 Inception Score: {inception_score:.4f}")
    
    # Test FID (without real data comparison)
    fid_score = evaluator.calculate_fid([], sample_transactions)
    print(f"📏 FID Score: {fid_score:.4f}")
    
    # Test full evaluation
    metrics = evaluator.evaluate_batch([], sample_transactions)
    print("\n📊 Full Evaluation Results:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n✅ Evaluation metrics test completed successfully!")

if __name__ == "__main__":
    test_evaluation_metrics() 