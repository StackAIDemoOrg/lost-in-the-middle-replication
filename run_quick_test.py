#!/usr/bin/env python3
"""
Quick end-to-end test of experiment_v2.py with minimal API calls.

Tests: 1 context length × 3 positions × 2 trials = 6 total API calls
"""

import json
import os
import sys
from experiment_v2 import (
    run_experiment,
    print_summary,
    plot_results,
    compute_cost,
    OPENAI_API_KEY
)


def main():
    """Run quick test experiment."""
    # Check API key
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)

    print("=" * 80)
    print("QUICK API TEST - EXPERIMENT V2")
    print("=" * 80)
    print("Configuration:")
    print("  - Context lengths: [8k tokens]")
    print("  - Positions: [0.1, 0.5, 0.9]")
    print("  - Trials per cell: 2")
    print("  - Total API calls: 6")
    print("=" * 80)
    print()

    # Run experiment with minimal config
    results = run_experiment(
        context_lengths_k=[8],
        positions=[0.1, 0.5, 0.9],
        trials_per_cell=2
    )

    # Save results
    output_file = "results_v2_test.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nTest results saved to: {output_file}")

    # Print summary
    print_summary(results)

    # Generate plots
    try:
        plot_results(results)
        print("\nTest plots generated successfully!")
    except Exception as e:
        print(f"\nNote: Plot generation skipped or failed: {e}")

    # Calculate cost
    total_cost = compute_cost(
        results["total_prompt_tokens"],
        results["total_completion_tokens"]
    )

    print("\n" + "=" * 80)
    print("QUICK TEST COMPLETE")
    print("=" * 80)
    print(f"Total cost: ${total_cost:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
