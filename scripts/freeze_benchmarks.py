#!/usr/bin/env python3
"""Script to generate and freeze benchmark evaluation sets.

This script generates static evaluation sets for SCAN and COGS benchmarks
using the Add-Jump and Subject-to-Object splits respectively. These sets
are frozen to ensure reproducibility across all training runs and evaluations.

The generated sets are:
- data/scan_id_eval.json: 2,000 in-distribution SCAN samples
- data/scan_ood_eval.json: 2,000 out-of-distribution SCAN samples (Add-Jump)
- data/cogs_id_eval.json: 2,000 in-distribution COGS samples
- data/cogs_ood_eval.json: 2,000 out-of-distribution COGS samples (Subject-to-Object)

Each file contains a list of {"input": "...", "output": "..."} pairs.
"""

import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hbar.benchmarks.scan_grammar import SCANGrammar
from hbar.benchmarks.cogs_grammar import COGSGrammar


def generate_scan_eval_sets(n_samples: int = 2000, seed: int = 42):
    """Generate SCAN evaluation sets with Add-Jump split.

    Args:
        n_samples: Number of samples for each set.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (id_samples, ood_samples).
    """
    print(f"Generating SCAN evaluation sets (n={n_samples}, seed={seed})...")

    grammar = SCANGrammar(seed=seed)
    train_pairs, test_pairs = grammar.get_add_jump_split(
        n_train=n_samples,
        n_test=n_samples,
    )

    id_samples = [{"input": cmd, "output": act} for cmd, act in train_pairs]
    ood_samples = [{"input": cmd, "output": act} for cmd, act in test_pairs]

    print(f"  ID samples: {len(id_samples)}")
    print(f"  OOD samples: {len(ood_samples)}")

    # Verify no 'jump' in compounds in ID set (except isolated 'jump')
    jump_compounds_id = [
        s for s in id_samples
        if "jump" in s["input"].lower() and s["input"].lower() != "jump"
    ]
    print(f"  ID samples with 'jump' in compound: {len(jump_compounds_id)} (should be 0)")

    # Verify all OOD samples contain 'jump' in compound
    jump_in_ood = [s for s in ood_samples if "jump" in s["input"].lower()]
    print(f"  OOD samples with 'jump': {len(jump_in_ood)} (should be {n_samples})")

    return id_samples, ood_samples


def generate_cogs_eval_sets(n_samples: int = 2000, seed: int = 42):
    """Generate COGS evaluation sets with Subject-to-Object split.

    Args:
        n_samples: Number of samples for each set.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (id_samples, ood_samples).
    """
    print(f"\nGenerating COGS evaluation sets (n={n_samples}, seed={seed})...")

    grammar = COGSGrammar(seed=seed)
    train_pairs, test_pairs = grammar.get_subject_object_split(
        n_train=n_samples,
        n_test=n_samples,
    )

    id_samples = [{"input": sent, "output": lf} for sent, lf in train_pairs]
    ood_samples = [{"input": sent, "output": lf} for sent, lf in test_pairs]

    print(f"  ID samples: {len(id_samples)}")
    print(f"  OOD samples: {len(ood_samples)}")

    # Verify biased nouns only in subject position in ID set
    # In ID set, biased nouns should be at the START of the sentence (subject)
    biased_in_obj_id = [
        s for s in id_samples
        for noun in grammar.BIASED_NOUNS
        if f" {noun} " in s["input"].lower()  # noun appears mid-sentence (object position)
    ]
    print(f"  ID samples with biased noun in object: {len(biased_in_obj_id)} (should be 0)")

    # Verify biased nouns only in object position in OOD set
    # In OOD set, biased nouns should NOT be at the start (they're objects)
    biased_in_subj_ood = [
        s for s in ood_samples
        for noun in grammar.BIASED_NOUNS
        if s["input"].lower().startswith(f"the {noun}") or s["input"].lower().startswith(f"a {noun}")
    ]
    print(f"  OOD samples with biased noun in subject: {len(biased_in_subj_ood)} (should be 0)")

    # Additional verification: check that biased nouns ARE in correct positions
    biased_in_subj_id = [
        s for s in id_samples
        for noun in grammar.BIASED_NOUNS
        if s["input"].lower().startswith(f"the {noun}") or s["input"].lower().startswith(f"a {noun}")
    ]
    print(f"  ID samples with biased noun in subject: {len(biased_in_subj_id)} (should be {n_samples})")

    biased_in_obj_ood = [
        s for s in ood_samples
        for noun in grammar.BIASED_NOUNS
        if f" {noun} " in s["input"].lower()  # noun appears mid-sentence (object position)
    ]
    print(f"  OOD samples with biased noun in object: {len(biased_in_obj_ood)} (should be {n_samples})")

    return id_samples, ood_samples


def save_eval_sets(output_dir: str = "data"):
    """Generate and save all evaluation sets.

    Args:
        output_dir: Directory to save the JSON files.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate SCAN sets
    scan_id, scan_ood = generate_scan_eval_sets()

    # Generate COGS sets
    cogs_id, cogs_ood = generate_cogs_eval_sets()

    # Save to JSON files
    files = {
        "scan_id_eval.json": scan_id,
        "scan_ood_eval.json": scan_ood,
        "cogs_id_eval.json": cogs_id,
        "cogs_ood_eval.json": cogs_ood,
    }

    print(f"\nSaving evaluation sets to {output_dir}/")
    for filename, data in files.items():
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Saved {filepath} ({len(data)} samples, {os.path.getsize(filepath):,} bytes)")

    print("\nDone! Evaluation sets are frozen for reproducibility.")


if __name__ == "__main__":
    save_eval_sets()
